from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime

from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def ovms_kserve_storage_uri_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    deployment_mode = request.param["deployment-mode"]

    isvc_kwargs = {
        "client": admin_client,
        "name": f"{request.param['name']}-{deployment_mode.lower()}",
        "namespace": model_namespace.name,
        "runtime": ovms_kserve_serving_runtime.name,
        "storage_uri": request.param["storage-uri"],
        "model_format": request.param["model-format"],
        "deployment_mode": request.param["deployment-mode"],
        "model_version": request.param["model-version"],
    }

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc
