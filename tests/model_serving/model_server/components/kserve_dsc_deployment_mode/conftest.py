from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.components.kserve_dsc_deployment_mode.utils import (
    patch_dsc_default_deployment_mode,
)
from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import ModelAndFormat
from utilities.jira import is_jira_open


@pytest.fixture(scope="class")
def skip_if_serverless_annotation_bug_present(
    admin_client: DynamicClient,
) -> None:
    jira_id = "RHOAIENG-19654"

    if is_jira_open(jira_id=jira_id, admin_client=admin_client):
        pytest.skip(reason=f"Bug {jira_id} is not fixed")


@pytest.fixture(scope="class")
def default_deployment_mode_in_dsc(
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    inferenceservice_config_cm: ConfigMap,
) -> Generator[DataScienceCluster, Any, Any]:
    yield from patch_dsc_default_deployment_mode(
        dsc_resource=dsc_resource,
        inferenceservice_config_cm=inferenceservice_config_cm,
        request_default_deployment_mode=request.param["default-deployment-mode"],
    )


@pytest.fixture(scope="class")
def inferenceservice_config_cm(admin_client: DynamicClient) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture(scope="class")
def patched_default_deployment_mode_in_dsc(
    request: FixtureRequest,
    default_deployment_mode_in_dsc: DataScienceCluster,
    inferenceservice_config_cm: ConfigMap,
) -> Generator[DataScienceCluster, Any, Any]:
    yield from patch_dsc_default_deployment_mode(
        dsc_resource=default_deployment_mode_in_dsc,
        inferenceservice_config_cm=inferenceservice_config_cm,
        request_default_deployment_mode=request.param["updated-deployment-mode"],
    )


@pytest.fixture(scope="class")
def ovms_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    openvino_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        runtime=openvino_kserve_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        model_version=request.param["model-version"],
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def restarted_inference_pod(ovms_inference_service: InferenceService) -> Pod:
    label_selector = (
        f"{ovms_inference_service.ApiGroup.SERVING_KSERVE_IO}/inferenceservice={ovms_inference_service.name}"
    )
    original_pod = [
        pod
        for pod in Pod.get(
            dyn_client=ovms_inference_service.client,
            namespace=ovms_inference_service.namespace,
            label_selector=label_selector,
        )
    ][0]

    original_pod.delete(wait=True)

    pods = [
        pod
        for pod in Pod.get(
            dyn_client=ovms_inference_service.client,
            namespace=ovms_inference_service.namespace,
            label_selector=label_selector,
        )
    ]

    if len(pods) != 1:
        raise ValueError(f"Expected 1 pod, got {len(pods)}")

    return pods[0]
