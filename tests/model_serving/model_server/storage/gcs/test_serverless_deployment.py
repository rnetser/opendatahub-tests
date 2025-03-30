import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.tensorflow import TENSORFLOW_INFERENCE_CONFIG

# pytestmark = [pytest.mark.serverless, pytest.mark.gcs, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, ovms_kserve_serving_runtime, ovms_kserve_storage_uri_inference_service",
    [
        pytest.param(
            {"name": "gcs-tensorflow-serverless"},
            {
                "runtime-name": "gcs-tensorflow-serverless",
                "supported-model-formats": [
                    {
                        "name": ModelFormat.TENSORFLOW,
                        "version": "0001",
                        "autoSelect": True,
                        "priority": 1,
                    }
                ],
            },
            {
                "name": ModelFormat.TENSORFLOW,
                "storage-uri": "gs://kfserving-examples/models/tensorflow/flowers",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "model-format": ModelFormat.TENSORFLOW,
                "model-version": "0001",
            },
        )
    ],
    indirect=True,
)
class TestTensorflowGcsServerless:
    def test_serverless_tensorflow_rest_inference(self, ovms_kserve_storage_uri_inference_service):
        """Verify that kserve Serverless tensorflow model can be queried using REST"""
        verify_inference_response(
            inference_service=ovms_kserve_storage_uri_inference_service,
            inference_config=TENSORFLOW_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
