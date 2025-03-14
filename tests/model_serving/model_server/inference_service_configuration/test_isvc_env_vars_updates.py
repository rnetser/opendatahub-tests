import pytest

from tests.model_serving.model_server.inference_service_configuration.constants import (
    ISVC_ENV_VARS,
)
from tests.model_serving.model_server.inference_service_configuration.utils import (
    verify_env_vars_in_isvc_pods,
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
    "model_env_variables": ISVC_ENV_VARS,
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
class TestRawISVCEnvVarsUpdatesSinglePod:
    def test_raw_with_isvc_env_vars(self, ovms_kserve_inference_service):
        """Test adding environment variables to the inference service"""
        verify_env_vars_in_isvc_pods(isvc=ovms_kserve_inference_service, env_vars=ISVC_ENV_VARS, vars_exist=True)

    def test_raw_remove_isvc_env_vars(self, removed_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        verify_env_vars_in_isvc_pods(isvc=removed_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=False)


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
                "min-replicas": 4,
            },
        )
    ],
    indirect=True,
)
class TestRawISVCEnvVarsUpdatesMultiPod:
    def test_raw_add_isvc_env_vars_multiple_pods(self, ovms_kserve_inference_service):
        """Test adding environment variables from the inference service"""
        verify_env_vars_in_isvc_pods(isvc=ovms_kserve_inference_service, env_vars=ISVC_ENV_VARS, vars_exist=False)

    def test_raw_remove_isvc_env_vars_multiple_pods(self, removed_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        verify_env_vars_in_isvc_pods(isvc=removed_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=False)


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
class TestServerlessISVCEnvVarsUpdatesSinglePod:
    def test_serverless_with_isvc_env_vars(self, ovms_kserve_inference_service):
        """Test adding environment variables to the inference service"""
        verify_env_vars_in_isvc_pods(isvc=ovms_kserve_inference_service, env_vars=ISVC_ENV_VARS, vars_exist=True)

    def test_serverless_remove_isvc_env_vars(self, removed_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        verify_env_vars_in_isvc_pods(isvc=removed_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=False)


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
                "min-replicas": 4,
            },
        )
    ],
    indirect=True,
)
class TestServerlessISVCEnvVarsUpdatesMultiPod:
    def test_serverless_add_isvc_env_vars_multiple_pods(self, ovms_kserve_inference_service):
        """Test adding environment variables from the inference service"""
        verify_env_vars_in_isvc_pods(isvc=ovms_kserve_inference_service, env_vars=ISVC_ENV_VARS, vars_exist=False)

    def test_serverless_remove_isvc_env_vars_multiple_pods(self, removed_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        verify_env_vars_in_isvc_pods(isvc=removed_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=False)
