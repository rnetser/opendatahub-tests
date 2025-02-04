import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelStoragePath,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG
from utilities.manifests.tensorflow import TENSORFLOW_INFERENCE_CONFIG

pytestmark = [pytest.mark.modelmesh, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, http_s3_ovms_model_mesh_serving_runtime, http_s3_openvino_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-authentication", "modelmesh-enabled": True},
            {"enable-auth": True, "enable-route": True},
            {"model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL},
        )
    ],
    indirect=True,
)
class TestModelMeshAuthentication:
    @pytest.mark.dependency(name="test_model_mesh_model_authentication_openvino_inference_with_tensorflow")
    def test_model_mesh_model_authentication_openvino_inference_with_tensorflow(
        self, http_s3_openvino_model_mesh_inference_service, http_model_mesh_inference_token
    ):
        """Verify model query with token using REST"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_model_mesh_inference_token,
        )

    @pytest.mark.dependency(name="test_model_mesh_disabled_model_authentication")
    def test_model_mesh_disabled_model_authentication(self, patched_remove_authentication_model_mesh_isvc):
        """Verify model query after authentication is disabled"""
        verify_inference_response(
            inference_service=patched_remove_authentication_model_mesh_isvc,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_model_mesh_disabled_model_authentication"])
    def test_model_mesh_re_enabled_model_authentication(
        self, http_s3_openvino_model_mesh_inference_service, http_model_mesh_inference_token
    ):
        """Verify model query after authentication is re-enabled"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_model_mesh_inference_token,
        )

    @pytest.mark.dependency(depends=["test_model_mesh_model_authentication_openvino_inference_with_tensorflow"])
    def test_model_mesh_model_authentication_using_invalid_token(self, http_s3_openvino_model_mesh_inference_service):
        """Verify model query with an invalid token"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token="dummy",
            authorized_user=False,
        )

    @pytest.mark.dependency(depends=["test_model_mesh_model_authentication_openvino_inference_with_tensorflow"])
    def test_model_mesh_model_authentication_without_token(self, http_s3_openvino_model_mesh_inference_service):
        """Verify model query without providing a token"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            authorized_user=False,
        )

    @pytest.mark.parametrize(
        "http_s3_ovms_external_route_model_mesh_serving_runtime, http_s3_openvino_second_model_mesh_inference_service",
        [
            pytest.param(
                {"enable-auth": True},
                {
                    "model-path": ModelStoragePath.TENSORFLOW_MODEL,
                    "model-format": ModelFormat.TENSORFLOW,
                    "runtime-fixture-name": "http_s3_ovms_external_route_model_mesh_serving_runtime",
                    "model-version": "2",
                },
            )
        ],
        indirect=True,
    )
    def test_model_mesh_block_cross_model_authentication(
        self,
        http_s3_ovms_external_route_model_mesh_serving_runtime,
        http_s3_openvino_model_mesh_inference_service,
        http_s3_openvino_second_model_mesh_inference_service,
        http_model_mesh_inference_token,
    ):
        """Verify model query with a second model's token is blocked"""
        verify_inference_response(
            inference_service=http_s3_openvino_second_model_mesh_inference_service,
            inference_config=TENSORFLOW_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_model_mesh_inference_token,
            authorized_user=False,
        )
