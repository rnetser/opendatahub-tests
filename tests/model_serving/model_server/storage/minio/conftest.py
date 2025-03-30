from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def ovms_minio_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    ovms_kserve_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        deployment_mode=request.param["deployment-mode"],
        model_format=request.param["model-format"],
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path=request.param["model-dir"],
        model_version=request.param["model-version"],
    ) as isvc:
        yield isvc
