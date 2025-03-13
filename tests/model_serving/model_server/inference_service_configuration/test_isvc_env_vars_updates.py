import pytest

from tests.model_serving.model_server.inference_service_configuration.constants import (
    ISVC_ENV_VARS,
)
from tests.model_serving.model_server.inference_service_configuration.utils import (
    verify_env_vars_in_isvc_pod,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    ModelVersion,
)

pytestmark = [pytest.mark.sanity, pytest.mark.usefixtures("valid_aws_config")]

RUNTIME_CONFIG = {
    "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
    "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
}
BASE_ISVC_CONFIG = {
    "name": "isvc-update",
    "model-version": ModelVersion.OPSET13,
    "model-dir": "test-dir",
}


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "raw-isvc-update"},
            RUNTIME_CONFIG,
            {
                **BASE_ISVC_CONFIG,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestRawISVCEnvVarsUpdates:
    @pytest.mark.parametrize(
        "patched_isvc_env_vars",
        [
            pytest.param({"action": "add-envs"}),
        ],
        indirect=True,
    )
    def test_raw_add_isvc_env_vars(self, patched_isvc_env_vars):
        """Test adding environment variables to the inference service"""
        verify_env_vars_in_isvc_pod(isvc=patched_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=True)

    @pytest.mark.parametrize(
        "patched_isvc_env_vars",
        [
            pytest.param({"action": "remove-envs"}),
        ],
        indirect=True,
    )
    def test_raw_remove_isvc_env_vars(self, patched_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        verify_env_vars_in_isvc_pod(isvc=patched_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=False)


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "serverless-isvc-update"},
            RUNTIME_CONFIG,
            {
                **BASE_ISVC_CONFIG,
                "deployment-mode": KServeDeploymentType.SERVERLESS,
            },
        )
    ],
    indirect=True,
)
class TestServerlessISVCEnvVarsUpdates:
    @pytest.mark.parametrize(
        "patched_isvc_env_vars",
        [
            pytest.param({"action": "add-envs"}),
        ],
        indirect=True,
    )
    def test_serverless_add_isvc_env_vars(self, patched_isvc_env_vars):
        """Test adding environment variables to the inference service"""
        verify_env_vars_in_isvc_pod(isvc=patched_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=True)

    @pytest.mark.parametrize(
        "patched_isvc_env_vars",
        [
            pytest.param({"action": "remove-envs"}),
        ],
        indirect=True,
    )
    def test_serverless_remove_isvc_env_vars(self, patched_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        verify_env_vars_in_isvc_pod(isvc=patched_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=False)
