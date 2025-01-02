import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelVersion,
    Protocols,
    ModelInferenceRuntime,
)
from utilities.inference_utils import Inference


pytestmark = pytest.mark.usefixtures("skip_if_no_deployed_openshift_serverless", "valid_aws_config")


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, openvino_kserve_serving_runtime, ovms_serverless_inference_service",
    [
        pytest.param(
            {"name": "kserve-serverless-onnx"},
            {"model-dir": "test-dir"},
            {
                "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
                "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
            },
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13},
        )
    ],
    indirect=True,
)
class TestONNXServerless:
    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-9045")
    def test_serverless_onnx_rest_inference(self, ovms_serverless_inference_service):
        """Verify that kserve Serverless ONNX model can be queried using REST"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            runtime=ModelInferenceRuntime.ONNX_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
