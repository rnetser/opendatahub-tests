import shlex
from typing import Dict

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import get_client
from ocp_resources.service_mesh_member import ServiceMeshMember
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config


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
def ci_s3_storage_uri(request) -> str:
    return f"s3://{py_config['ci_s3_bucket_name']}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def model_pvc(admin_client: DynamicClient, model_namespace: Namespace) -> PersistentVolumeClaim:
    with PersistentVolumeClaim(
        name="model-pvc",
        namespace=model_namespace.name,
        client=admin_client,
        size="15Gi",
        accessmodes="ReadWriteOnce",
    ) as pvc:
        # pvc.wait_for_status(status=pvc.Status.BOUND, timeout=60)
        yield pvc


@pytest.fixture(scope="class")
def downloaded_model_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ci_s3_storage_uri: str,
    model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key: str,
) -> str:
    mount_path: str = "data"
    model_dir: str = "model-dir"
    containers = [
        {
            "name": "model-downloader",
            "image": "quay.io/redhat_msi/qe-tools-base-image",
            "args": [
                "sh",
                "-c",
                "sleep infinity",
            ],
            "env": [
                {"name": "AWS_ACCESS_KEY_ID", "value": aws_access_key},
                {"name": "AWS_SECRET_ACCESS_KEY", "value": aws_secret_access_key},
            ],
            "volumeMounts": [{"mountPath": mount_path, "name": model_pvc.name, "subPath": model_dir}],
        }
    ]
    volumes = [{"name": model_pvc.name, "persistentVolumeClaim": {"claimName": model_pvc.name}}]

    with Pod(
        client=admin_client,
        namespace=model_namespace.name,
        name="download-model-data",
        containers=containers,
        volumes=volumes,
    ) as pod:
        pod.wait_for_status(status=Pod.Status.RUNNING)
        pod.execute(command=shlex.split(f"aws s3 cp --recursive {ci_s3_storage_uri} /{mount_path}/{model_dir}"))

    return model_dir


@pytest.fixture(scope="class")
def serving_runtime(
    request,
    admin_client: DynamicClient,
    service_mesh_member,
    model_namespace: Namespace,
    downloaded_model_data: str,
) -> ServingRuntime:
    containers = [
        {
            "name": "kserve-container",
            "image": "quay.io/modh/openvino_model_server:stable",
            "args": [
                f"--model_name={request.param['name']}",
                f"--model_path=/mnt/models/{downloaded_model_data}",
                "--rest_port=8888",
                "--grpc_bind_address=0.0.0.0",
            ],
            "ports": [{"containerPort": 8888, "protocol": "TCP"}],
        }
    ]

    with ServingRuntime(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        annotations={"opendatahub.io/apiProtocol": "REST"},
        containers=containers,
        supported_model_formats=[
            {
                "name": request.param["model-name"],
                "autoselect": "true",
                "version": request.param["model-version"],
            },
        ],
        multi_model=request.param["multi-model"],
        protocol_versions=["v2", "grpc-v2"],
    ) as mlserver:
        yield mlserver


@pytest.fixture(scope="class")
def inference_service(
    request,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    model_pvc: PersistentVolumeClaim,
    downloaded_model_data: str,
) -> InferenceService:
    with InferenceService(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        annotations=INFERENCE_ANNOTATIONS,
        predictor={
            "model": {
                "modelFormat": {"name": serving_runtime.instance.spec.supportedModelFormats[0].name},
                "version": "1",
                "runtime": serving_runtime.name,
                "storageUri": f"pvc://{model_pvc.name}/{downloaded_model_data}",
            },
        },
    ) as inference_service:
        inference_service.wait_for_condition(
            condition=inference_service.Condition.READY,
            status=inference_service.Condition.Status.TRUE,
            timeout=10 * 60,
        )
        yield inference_service
