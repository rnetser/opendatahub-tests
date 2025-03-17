import pytest

from tests.model_serving.model_server.inference_service_configuration.constants import (
    BASE_ISVC_CONFIG,
    RUNTIME_CONFIG,
)
from tests.model_serving.model_server.inference_service_configuration.utils import (
    wait_for_new_running_inference_pods,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.sanity, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "raw-isvc-replicas"},
            RUNTIME_CONFIG,
            {
                **BASE_ISVC_CONFIG,
                "min-replicas": 2,
                "max-replicas": 4,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestRawISVCReplicasUpdates:
    @pytest.mark.dependency(name="test_raw_increase_isvc_replicas")
    def test_raw_increase_isvc_replicas(self, isvc_pods, patched_isvc_replicas):
        """Test replicas increase"""
        wait_for_new_running_inference_pods(isvc=patched_isvc_replicas, orig_pods=isvc_pods, expected_num_pods=2)

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
    def test_raw_decrease_isvc_replicas(self, isvc_pods, patched_isvc_replicas):
        """Test replicas decrease"""
        wait_for_new_running_inference_pods(isvc=patched_isvc_replicas, orig_pods=isvc_pods, expected_num_pods=2)

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
