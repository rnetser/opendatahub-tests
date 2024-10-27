import base64
from typing import Optional

from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from utilities.inference_utils import Inference


LOGGER = get_logger(name=__name__)


def base64_encode_str(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def verify_inference_response(
    inference_service: InferenceService,
    runtime: str,
    inference_type: str,
    protocol: str,
    model_name: str,
    inference_text: str,
    expected_response_text: str,
    insecure: bool = True,
    token: Optional[str] = None,
) -> None:
    inference = Inference(
        inference_service=inference_service,
        runtime=runtime,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference(
        model_name=model_name,
        text=inference_text,
        token=token,
        insecure=insecure,
    )

    try:
        assert res[inference.inference_response_text_key_name] == expected_response_text

    except KeyError:
        LOGGER.error(f"Inference response text not found in response. Response: {res}")
        raise
