import pytest

from tests.model_serving.model_server.authentication.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, ModelFormat, ModelStoragePath, Protocols, RuntimeQueryKeys
from utilities.inference_utils import Inference

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, http_s3_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL},
            {"deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT},
        )
    ],
    indirect=True,
)
class TestRawDeployment:
    def test_rest_raw_deployment_internal_route(self, http_s3_inference_service):
        verify_inference_response(
            inference_service=http_s3_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_isvc_visibility_annotation",
        [
            pytest.param(
                {"visibility": "cluster-local"},
            )
        ],
        indirect=True,
    )
    def test_rest_raw_deployment_external_route(self, patched_isvc_visibility_annotation):
        verify_inference_response(
            inference_service=patched_isvc_visibility_annotation,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
