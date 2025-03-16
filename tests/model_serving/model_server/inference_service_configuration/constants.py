from utilities.constants import ModelFormat, ModelInferenceRuntime, ModelVersion

RUNTIME_CONFIG = {
    "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
    "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
}
BASE_ISVC_CONFIG = {
    "name": "isvc-replicas",
    "model-version": ModelVersion.OPSET13,
    "model-dir": "test-dir",
}

ISVC_ENV_VARS = [
    {"name": "TEST_ENV_VAR1", "value": "test_value1"},
    {"name": "TEST_ENV_VAR2", "value": "test_value2"},
]
