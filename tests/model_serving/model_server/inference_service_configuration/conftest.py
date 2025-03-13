from typing import Any, Generator

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor

from utilities.constants import Containers


@pytest.fixture
def patched_isvc_env_vars(
    request: pytest.FixtureRequest, s3_models_inference_service: InferenceService
) -> Generator[InferenceService, Any, Any]:
    env_vars = [
        {"name": "TEST_ENV_VAR1", "value": "test_value1"},
        {"name": "TEST_ENV_VAR2", "value": "test_value2"},
    ]

    isvc_spec = s3_models_inference_service.instance.spec
    action = request.param["action"]

    for container in isvc_spec:
        if container.name == Containers.KSERVE_CONTAINER_NAME:
            if action == "add-envs":
                container.env.extend(env_vars)

            elif action == "remove-envs":
                for env_var in env_vars:
                    container.env = [env_var for env_var in container.env if env_var not in env_vars]

            else:
                raise ValueError(f"Invalid action: {action}")

    with ResourceEditor(patches={s3_models_inference_service: {"spec": isvc_spec}}):
        yield s3_models_inference_service
