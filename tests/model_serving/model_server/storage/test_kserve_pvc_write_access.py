import pytest


pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, storage_uri, serving_runtime, inference_service",
    [
        pytest.param(
            {"name": "pvc-write-access"},
            {"model-dir": "test-dir"},
            {
                "name": "ovms-runtime",
                "model-name": "onnx",
                "model-version": "1",
                "multi-model": False,
            },
            {"name": "onnx", "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestKservePVCWriteAccess:
    def test_isvc_read_only_annotation_false(self, inference_service):
        pass

    def test_isvc_read_only_annotation_true(self, inference_service):
        pass

    def test_isvc_read_only_annotation_default_value(self, inference_service):
        pass

    def test_isvc_modified_read_only_annotation(self, inference_service):
        pass
