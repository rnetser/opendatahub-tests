import pytest

from tests.model_serving.model_server.inference_service_configuration.constants import (
    ISVC_ENV_VARS,
)
from tests.model_serving.model_server.inference_service_configuration.utils import (
    verify_env_vars_in_isvc_pods,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.sanity, pytest.mark.usefixtures("valid_aws_config")]

RUNTIME_CONFIG = {
    "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
    "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
}
BASE_ISVC_CONFIG = {
    "name": "isvc-replicas",
    "model-version": ModelVersion.OPSET13,
    "model-dir": "test-dir",
}


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "raw-isvc-replicas"},
            RUNTIME_CONFIG,
            {
                **BASE_ISVC_CONFIG,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestRawISVCReplicasUpdates:
    @pytest.mark.parametrize(
        "patched_isvc_replicas",
        [
            pytest.param({"min-replicas": 2, "max-replicas": 4}),
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_raw_increase_isvc_replicas")
    def test_raw_increase_isvc_replicas(self, patched_isvc_replicas):
        """Test replicas increase"""
        verify_env_vars_in_isvc_pods(isvc=patched_isvc_replicas, env_vars=ISVC_ENV_VARS, vars_exist=True)

    @pytest.mark.dependency(depends=["test_raw_increase_isvc_replicas"])
    def test_raw_increase_isvc_replicas_inference(self, ovms_kserve_inference_service):
        """Verify inference after replicas increase"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_isvc_replicas",
        [
            pytest.param({"min-replicas": 1, "max-replicas": 1}),
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_raw_decrease_isvc_replicas")
    def test_raw_decrease_isvc_replicas(self, patched_isvc_replicas):
        """Test replicas decrease"""
        verify_env_vars_in_isvc_pods(isvc=patched_isvc_replicas, env_vars=ISVC_ENV_VARS, vars_exist=False)

    @pytest.mark.dependency(depends=["test_raw_decrease_isvc_replicas"])
    def test_raw_decrease_isvc_replicas_inference(self, ovms_kserve_inference_service):
        """Verify inference after replicas decrease"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "serverless-isvc-replicas"},
            RUNTIME_CONFIG,
            {
                **BASE_ISVC_CONFIG,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestServerlessISVCReplicasUpdates:
    @pytest.mark.parametrize(
        "patched_isvc_replicas",
        [
            pytest.param({"min-replicas": 2, "max-replicas": 4}),
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_serverless_increase_isvc_replicas")
    def test_serverless_increase_isvc_replicas(self, patched_isvc_replicas):
        """Test replicas increase"""
        verify_env_vars_in_isvc_pods(isvc=patched_isvc_replicas, env_vars=ISVC_ENV_VARS, vars_exist=True)

    @pytest.mark.dependency(depends=["test_serverless_increase_isvc_replicas"])
    def test_serverless_increase_isvc_replicas_inference(self, ovms_kserve_inference_service):
        """Verify inference after replicas increase"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_isvc_replicas",
        [
            pytest.param({"min-replicas": 1, "max-replicas": 1}),
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_serverless_decrease_isvc_replicas")
    def test_serverless_decrease_isvc_replicas(self, patched_isvc_replicas):
        """Test replicas decrease"""
        verify_env_vars_in_isvc_pods(isvc=patched_isvc_replicas, env_vars=ISVC_ENV_VARS, vars_exist=False)

    @pytest.mark.dependency(depends=["test_serverless_decrease_isvc_replicas"])
    def test_serverless_decrease_isvc_replicas_inference(self, ovms_kserve_inference_service):
        """Verify inference after replicas decrease"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
