from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.multi_node.utils import (
    delete_multi_node_pod_by_role,
)
from utilities.constants import KServeDeploymentType, StorageClassName
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc
from utilities.infra import (
    create_ns,
    get_pods_by_isvc_label,
    wait_for_inference_deployment_replicas,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def nodes(admin_client: DynamicClient) -> list[Node]:
    return list(Node.get(dyn_client=admin_client))


@pytest.fixture(scope="session")
def nvidia_gpu_nodes(nodes: list[Node]) -> list[Node]:
    return [node for node in nodes if "nvidia.com/gpu.present" in node.labels.keys()]


@pytest.fixture(scope="session")
def skip_if_no_gpu_nodes(nvidia_gpu_nodes: list[Node]) -> None:
    if len(nvidia_gpu_nodes) < 2:
        pytest.skip("Multi-node tests can only run on a Cluster with at least 2 GPU Worker nodes")


@pytest.fixture(scope="module")
def model_namespace_scope_module(
    request: FixtureRequest, admin_client: DynamicClient
) -> Generator[Namespace, Any, Any]:
    with create_ns(admin_client=admin_client, name=request.param["name"]) as ns:
        yield ns


@pytest.fixture(scope="module")
def model_pvc_scope_module(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace_scope_module: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        client=admin_client,
        name="model-pvc",
        namespace=model_namespace_scope_module.name,
        size="40Gi",
        accessmodes="ReadWriteMany",
        storage_class=StorageClassName.NFS,
    ) as pvc:
        pvc.wait_for_status(status=pvc.Status.BOUND, timeout=120)
        yield pvc


@pytest.fixture(scope="module")
def models_bucket_downloaded_model_data_scope_module(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace_scope_module: Namespace,
    models_s3_bucket_name: str,
    model_pvc_scope_module: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    return download_model_data(
        admin_client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace_scope_module.name,
        model_pvc_name=model_pvc_scope_module.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
    )


@pytest.fixture(scope="class")
def multi_node_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace_scope_module: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-multinode-runtime",  # TODO: rename servingruntime when RHOAIENG-16147 is resolved
        namespace=model_namespace_scope_module.name,
        template_name="vllm-multinode-runtime-template",
        multi_model=False,
        enable_http=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def multi_node_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace_scope_module: Namespace,
    multi_node_serving_runtime: ServingRuntime,
    model_pvc_scope_module: PersistentVolumeClaim,
    models_bucket_downloaded_model_data_scope_module: str,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace_scope_module.name,
        runtime=multi_node_serving_runtime.name,
        storage_uri=f"pvc://{model_pvc_scope_module.name}/{models_bucket_downloaded_model_data_scope_module}",
        model_format=multi_node_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        autoscaler_mode="external",
        multi_node_worker_spec={},
        wait_for_predictor_pods=False,
    ) as isvc:
        wait_for_inference_deployment_replicas(
            client=admin_client,
            isvc=isvc,
            expected_num_deployments=2,
            runtime_name=multi_node_serving_runtime.name,
        )
        yield isvc


@pytest.fixture(scope="class")
def multi_node_predictor_pods_scope_class(
    admin_client: DynamicClient,
    multi_node_inference_service: InferenceService,
) -> list[Pod]:
    return get_pods_by_isvc_label(
        client=admin_client,
        isvc=multi_node_inference_service,
    )


@pytest.fixture(scope="function")
def patched_multi_node_isvc_external_route(
    multi_node_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            multi_node_inference_service: {
                "metadata": {"labels": {"networking.kserve.io/visibility": "exposed"}},
            }
        }
    ):
        yield multi_node_inference_service


@pytest.fixture()
def ray_ca_tls_secret(admin_client: DynamicClient) -> Secret:
    return Secret(
        client=admin_client,
        name="ray-ca-tls",
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture()
def ray_tls_secret(admin_client: DynamicClient, multi_node_inference_service: InferenceService) -> Secret:
    return Secret(
        client=admin_client,
        name="ray-tls",
        namespace=multi_node_inference_service.namespace,
    )


@pytest.fixture()
def deleted_serving_runtime(
    multi_node_serving_runtime: ServingRuntime,
) -> Generator[None, Any, None]:
    multi_node_serving_runtime.clean_up()

    yield

    multi_node_serving_runtime.deploy()


@pytest.fixture()
def deleted_multi_node_pod(
    request: FixtureRequest,
    admin_client: DynamicClient,
    multi_node_inference_service: InferenceService,
) -> None:
    delete_multi_node_pod_by_role(
        client=admin_client,
        isvc=multi_node_inference_service,
        role=request.param["pod-role"],
    )
