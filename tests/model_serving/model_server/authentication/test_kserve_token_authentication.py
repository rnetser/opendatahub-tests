import re

import pytest

from tests.model_serving.model_server.authentication.utils import (
    verify_inference_response,
)
from utilities.inference_utils import INFERENCE_QUERIES, Inference

pytestmark = pytest.mark.usefixtures("skip_if_no_authorino_operator", "valid_aws_config")


CAIKIT_STR: str = "caikit"
CAIKIT_TGIS_RUNTIME_STR: str = f"{CAIKIT_STR}-tgis-runtime"
INFERENCE_QUERY = INFERENCE_QUERIES["nitrogen-boil-temp"]
FLAN_MODEL_NAME: str = f"flan-t5-small-{CAIKIT_STR}"


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, s3_serving_runtime, s3_inference_service",
    [
        pytest.param(
            {"name": "kserve-token-authentication"},
            {"model-dir": f"flan-t5-small/{FLAN_MODEL_NAME}"},
            {
                "name": CAIKIT_TGIS_RUNTIME_STR,
                "model-name": CAIKIT_STR,
                "template-name": "caikit-tgis-serving-template",
                "multi-model": False,
            },
            {
                "name": CAIKIT_STR,
                "deployment-mode": "Serverless",
                "enable-model-auth": True,
            },
        )
    ],
    indirect=True,
)
class TestKserveTokenAuthentication:
    def test_model_authentication_using_rest(self, s3_serving_runtime, s3_inference_service, inference_token):
        verify_inference_response(
            inference_service=s3_inference_service,
            runtime=s3_serving_runtime.name,
            inference_type=Inference.ALL_TOKENS,
            protocol="http",
            model_name=CAIKIT_STR,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
            token=inference_token,
        )

    def test_model_authentication_using_grpc(self, s3_serving_runtime, s3_inference_service, inference_token):
        verify_inference_response(
            inference_service=s3_inference_service,
            runtime=s3_serving_runtime.name,
            inference_type=Inference.STREAMING,
            protocol="grpc",
            model_name=FLAN_MODEL_NAME,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
            token=inference_token,
        )

    @pytest.mark.dependency(name="test_disabled_model_authentication")
    def test_disabled_model_authentication(self, s3_serving_runtime, patched_remove_authentication_isvc):
        verify_inference_response(
            inference_service=patched_remove_authentication_isvc,
            runtime=s3_serving_runtime.name,
            inference_type=Inference.ALL_TOKENS,
            protocol="http",
            model_name=CAIKIT_STR,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
        )

    @pytest.mark.dependency(depends=["test_disabled_model_authentication"])
    def test_re_enabled_model_authentication(self, s3_serving_runtime, s3_inference_service, inference_token):
        verify_inference_response(
            inference_service=s3_inference_service,
            runtime=s3_serving_runtime.name,
            inference_type=Inference.ALL_TOKENS,
            protocol="http",
            model_name=CAIKIT_STR,
            inference_text=INFERENCE_QUERY["text"],
            expected_response_text=INFERENCE_QUERY["response_text"],
            token=inference_token,
        )

    def test_model_authentication_using_invalid_token(self, s3_serving_runtime, s3_inference_service):
        inference = Inference(
            inference_service=s3_inference_service,
            runtime=s3_serving_runtime.name,
            inference_type=Inference.ALL_TOKENS,
            protocol="http",
        )
        out = inference.run_inference(
            model_name=CAIKIT_STR,
            text=INFERENCE_QUERY["text"],
            insecure=True,
            token="dummy",
        )

        if auth_reason := re.search(r"x-ext-auth-reason: (.*)", out["output"], re.MULTILINE):
            assert auth_reason.group(1) == "not authenticated"


# class TestKserveCrossModelAuthentication:
#     def test_block_cross_model_authentication(self):
#     pass
