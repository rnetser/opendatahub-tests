from typing import Dict

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.service_mesh_member import ServiceMeshMember
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.storage.utils import base64_encode_str

TRANSFORMERS_CACHE_ENV_VAR: Dict[str, str] = {
    "name": "TRANSFORMERS_CACHE",
    "value": "/tmp/transformers_cache",
}
INFERENCE_ANNOTATIONS: Dict[str, str] = {
    "serving.knative.openshift.io/enablePassthrough": "true",
    "sidecar.istio.io/inject": "true",
    "sidecar.istio.io/rewriteAppHTTPProbers": "true",
}
SMM_SPEC: Dict[str, str] = {"name": "data-science-smcp", "namespace": "istio-system"}


@pytest.fixture(scope="session")
def admin_client() -> DynamicClient:
    return get_client()


@pytest.fixture(scope="class")
def model_namespace(request, admin_client: DynamicClient) -> Namespace:
    with Namespace(
        client=admin_client,
        name=request.param["name"],
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns


@pytest.fixture(scope="class")
def service_mesh_member(admin_client: DynamicClient, model_namespace: Namespace) -> ServiceMeshMember:
    with ServiceMeshMember(
        client=admin_client,
        name="default",
        namespace=model_namespace.name,
        control_plane_ref=SMM_SPEC,
    ) as smm:
        yield smm


@pytest.fixture(scope="class")
def storage_uri(request) -> str:
    return f"s3://{py_config['model_s3_bucket_name']}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key: str,
    aws_secret_access_key: str,
) -> Secret:
    data = {
        "AWS_ACCESS_KEY_ID": base64_encode_str(text=aws_access_key),
        "AWS_SECRET_ACCESS_KEY": base64_encode_str(text=aws_secret_access_key),
        "AWS_S3_BUCKET": base64_encode_str(text=py_config["model_s3_bucket_name"]),
        "AWS_S3_ENDPOINT": base64_encode_str(text=py_config["model_s3_endpoint"]),
    }
    name = model_namespace.name

    with Secret(
        client=admin_client,
        namespace=name,
        name=name,
        data_dict=data,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def model_service_account(admin_client: DynamicClient, endpoint_s3_secret: Namespace) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def serving_runtime(
    request,
    admin_client: DynamicClient,
    service_mesh_member,
    model_namespace: Namespace,
) -> ServingRuntime:
    containers = [
        {
            "name": "kserve-container",
            "image": "quay.io/modh/openvino_model_server:stable",
            "args": [f"--model-name={request.param['model-name']}", "model_path=/mnt/models"],
            "env": [TRANSFORMERS_CACHE_ENV_VAR],
            "ports": [{"containerPort": 8888, "protocol": "TCP"}],
        }
    ]

    with ServingRuntime(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        containers=containers,
        supported_model_formats=[
            {"name": request.param["model-name"], "autoselect": "true"},
        ],
        multi_model=request.param["multi-model"],
    ) as mlserver:
        yield mlserver


@pytest.fixture(scope="class")
def inference_service(
    request,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    endpoint_s3_secret: Secret,
    storage_uri: str,
    model_service_account: ServiceAccount,
) -> InferenceService:
    with InferenceService(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        annotations=INFERENCE_ANNOTATIONS,
        predictor={
            "model": {
                "modelFormat": {"name": serving_runtime.instance.spec.supportedModelFormats[0].name},
                "runtime": serving_runtime.name,
                "storageUri": storage_uri,
            },
            "serviceAccountName": model_service_account.name,
        },
    ) as inference_service:
        inference_service.wait_for_condition(
            condition=inference_service.Condition.READY,
            status=inference_service.Condition.Status.TRUE,
            timeout=10 * 60,
        )
        yield inference_service
