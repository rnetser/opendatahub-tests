from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.components.kserve_dsc_deployment_mode.utils import (
    wait_for_default_deployment_mode_in_cm,
)
from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import ModelAndFormat
from utilities.jira import is_jira_open


@pytest.fixture(scope="class")
def skip_if_serverless_annotation_bug_present(
    admin_client: DynamicClient,
) -> None:
    jira_id = "RHOAIENG-16954"

    if is_jira_open(jira_id=jira_id, admin_client=admin_client):
        pytest.skip(reason=f"Bug {jira_id} is not fixed")


@pytest.fixture(scope="class")
def default_deployment_mode_in_dsc(
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    inferenceservice_config_cm: ConfigMap,
) -> Generator[DataScienceCluster, Any, Any]:
    request_default_deployment_mode: str = request.param["default-deployment-mode"]

    with ResourceEditor(
        patches={
            dsc_resource: {
                "spec": {"components": {"kserve": {"defaultDeploymentMode": request_default_deployment_mode}}}
            }
        }
    ):
        wait_for_default_deployment_mode_in_cm(
            config_map=inferenceservice_config_cm, deployment_mode=request_default_deployment_mode
        )
        yield dsc_resource


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
    request_deployment_mode: str = request.param["updated-deployment-mode"]

    with ResourceEditor(
        patches={
            default_deployment_mode_in_dsc: {
                "spec": {"components": {"kserve": {"defaultDeploymentMode": request_deployment_mode}}}
            }
        }
    ):
        wait_for_default_deployment_mode_in_cm(
            config_map=inferenceservice_config_cm, deployment_mode=request_deployment_mode
        )
        yield default_deployment_mode_in_dsc


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
