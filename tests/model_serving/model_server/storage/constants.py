from typing import Any, Dict

KSERVE_CONTAINER_NAME: str = "kserve-container"

KSERVE_OVMS_SERVING_RUNTIME_PARAMS: Dict[str, Any] = {
    "name": "ovms-runtime",
    "model-name": "onnx",
    "template-name": "kserve-ovms",
    "model-version": "1",
    "multi-model": False,
}

INFERENCE_SERVICE_PARAMS = {"name": "onnx"}
