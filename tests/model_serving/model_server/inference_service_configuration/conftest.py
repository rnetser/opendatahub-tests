from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.inference_service_configuration.constants import ISVC_ENV_VARS
from tests.model_serving.model_server.inference_service_configuration.utils import wait_for_new_deployment_generation


@pytest.fixture
def patched_isvc_env_vars(
    request: pytest.FixtureRequest, admin_client: DynamicClient, ovms_kserve_inference_service: InferenceService
) -> Generator[InferenceService, Any, Any]:
    isvc_predictor_spec_model_env = ovms_kserve_inference_service.instance.spec.predictor.model.get("env", [])
    action = request.param["action"]

    if action == "add-envs":
        isvc_predictor_spec_model_env.extend(ISVC_ENV_VARS)

    elif action == "remove-envs":
        isvc_predictor_spec_model_env = [
            env_var for env_var in isvc_predictor_spec_model_env if env_var not in ISVC_ENV_VARS
        ]

    else:
        raise ValueError(f"Invalid action: {action}")

    with ResourceEditor(
        patches={
            ovms_kserve_inference_service: {"spec": {"predictor": {"model": {"env": isvc_predictor_spec_model_env}}}}
        }
    ):
        wait_for_new_deployment_generation(client=admin_client, isvc=ovms_kserve_inference_service)

        yield ovms_kserve_inference_service
