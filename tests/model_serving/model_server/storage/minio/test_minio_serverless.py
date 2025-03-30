import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, MinIo, ModelFormat, Protocols
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


pytestmark = [pytest.mark.serverless, pytest.mark.minio, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, ovms_kserve_serving_runtime, ovms_minio_inference_service",
    [
        pytest.param(
            {"name": "minio-serverless"},
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
            {
                "runtime-name": "minio-ovms",
                "model-format": {"name": ModelFormat.ONNX, "version": "1"},
            },
            {
                "name": "loan-model",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "model-format": ModelFormat.ONNX,
                "model-version": "1",
                "model-dir": "onnx/mnist",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestMinioServerless:
    def test_minio_serverless_inference(
        self,
        ovms_minio_inference_service,
    ) -> None:
        """Verify that kserve Serverless minio model can be queried using REST"""
        verify_inference_response(
            inference_service=ovms_minio_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=f"infer-{ModelFormat.MNIST}",
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
