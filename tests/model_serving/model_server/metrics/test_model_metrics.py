import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelInferenceRuntime, ModelStoragePath, Protocols
from utilities.inference_utils import Inference
from utilities.monitoring import validate_metrics_value

pytestmark = pytest.mark.usefixtures("skip_if_no_deployed_openshift_serverless", "valid_aws_config")


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "kserve-token-authentication"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL},
        )
    ],
    indirect=True,
)
class TestModelMetrics:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_success_requests(self, http_s3_caikit_serverless_inference_service, prometheus):
        """Verify number of successful model requests in OpenShift monitoring system (UserWorkloadMonitoring)metrics"""
        verify_inference_response(
            inference_service=http_s3_caikit_serverless_inference_service,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
        validate_metrics_value(
            prometheus=prometheus,
            metric_name="tgi_request_count",
            expected_value="1",
        )

    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2555")
    def test_model_metrics_num_total_requests(self, http_s3_caikit_serverless_inference_service, prometheus):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring)metrics"""
        verify_inference_response(
            inference_service=http_s3_caikit_serverless_inference_service,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
