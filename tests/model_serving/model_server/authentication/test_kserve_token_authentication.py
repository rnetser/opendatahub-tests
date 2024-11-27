import pytest

from tests.model_serving.model_server.authentication.utils import (
    verify_inference_response,
)
from utilities.constants import ModelFormat, Protocols, RuntimeQueryKeys
from utilities.inference_utils import INFERENCE_QUERIES, Inference

pytestmark = pytest.mark.usefixtures("skip_if_no_authorino_operator", "valid_aws_config")

INFERENCE_QUERY = INFERENCE_QUERIES["nitrogen-boil-temp"]
FLAN_MODEL_NAME: str = f"flan-t5-small-{ModelFormat.CAIKIT}"


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "kserve-token-authentication"},
            {"model-dir": f"flan-t5-small/{FLAN_MODEL_NAME}"},
        )
    ],
    indirect=True,
)
class TestKserveTokenAuthentication:
    @pytest.mark.smoke
    @pytest.mark.dependency(name="test_model_authentication_using_rest")
    def test_model_authentication_using_rest(self, http_s3_inference_service, http_inference_token):
        # Verify model query with token using REST
        verify_inference_response(
            inference_service=http_s3_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
            token=http_inference_token,
        )

    @pytest.mark.smoke
    def test_model_authentication_using_grpc(self, grpc_s3_inference_service, grpc_inference_token):
        # Verify model query with token using GRPC
        verify_inference_response(
            inference_service=grpc_s3_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.STREAMING,
            protocol=Protocols.GRPC,
            model_name=FLAN_MODEL_NAME,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
            token=grpc_inference_token,
        )

    @pytest.mark.dependency(name="test_disabled_model_authentication")
    def test_disabled_model_authentication(self, patched_remove_authentication_isvc):
        # Verify model query after authentication is disabled
        verify_inference_response(
            inference_service=patched_remove_authentication_isvc,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
        )

    @pytest.mark.dependency(depends=["test_disabled_model_authentication"])
    def test_re_enabled_model_authentication(self, http_s3_inference_service, http_inference_token):
        # Verify model query after authentication is re-enabled
        verify_inference_response(
            inference_service=http_s3_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
            token=http_inference_token,
        )

    def test_model_authentication_using_invalid_token(self, http_s3_inference_service):
        # Verify model query with an invalid token
        verify_inference_response(
            inference_service=http_s3_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            inference_text=INFERENCE_QUERY["text"],
            token="dummy",
            authorized_user=False,
        )

    def test_model_authentication_without_token(self, http_s3_inference_service):
        # Verify model query without providing a token
        verify_inference_response(
            inference_service=http_s3_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            inference_text=INFERENCE_QUERY["text"],
            authorized_user=False,
        )

    @pytest.mark.sanity
    def test_block_cross_model_authentication(self, http_s3_inference_service, grpc_inference_token):
        # Verify model query with a second model's token is blocked
        verify_inference_response(
            inference_service=http_s3_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            inference_text=INFERENCE_QUERY["text"],
            token=grpc_inference_token,
            authorized_user=False,
        )
