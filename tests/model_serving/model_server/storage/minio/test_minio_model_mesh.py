import pytest

from tests.model_serving.model_server.storage.minio.constants import (
    INFERENCE_TYPE,
    MINIO_DATA_CONNECTION_CONFIG,
    MINIO_INFERENCE_CONFIG,
    MINIO_RUNTIME_CONFIG,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    MinIo,
    Protocols,
)
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.modelmesh, pytest.mark.minio, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, http_s3_ovms_model_mesh_serving_runtime, "
    "model_mesh_ovms_minio_inference_service",
    [
        pytest.param(
            {
                "name": f"{MinIo.Metadata.NAME}-{KServeDeploymentType.MODEL_MESH.lower()}",
                "modelmesh-enabled": True,
            },
            MinIo.PodConfig.KSERVE_MINIO_CONFIG,
            MINIO_DATA_CONNECTION_CONFIG,
            MINIO_RUNTIME_CONFIG,
            {
                "deployment-mode": KServeDeploymentType.MODEL_MESH,
                **MINIO_INFERENCE_CONFIG,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestMinioModelMesh:
    def test_minio_model_mesh_inference(
        self,
        model_mesh_ovms_minio_inference_service,
    ) -> None:
        """Verify that model mesh minio model can be queried using REST"""
        verify_inference_response(
            inference_service=model_mesh_ovms_minio_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=INFERENCE_TYPE,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
