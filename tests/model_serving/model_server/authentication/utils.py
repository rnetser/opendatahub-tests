import base64
import re
from typing import Dict, Optional

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
    expected_response_text: Optional[str] = "",
    insecure: bool = True,
    token: Optional[str] = None,
    authorized_user: Optional[bool] = None,
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

    if authorized_user is False:
        if auth_reason := re.search(r"x-ext-auth-reason: (.*)", res["output"], re.MULTILINE):
            assert auth_reason.group(1) == "not authenticated"

    if inference.inference_response_text_key_name:
        assert res["output"][inference.inference_response_text_key_name] == expected_response_text

    elif output := re.findall(r"generated_text\": \"(.*)\"", res["output"], re.MULTILINE):
        assert "".join(output) == expected_response_text

    else:
        LOGGER.error(f"Inference response text not found in response. Response: {res}")
        raise


def get_s3_secret_dict(
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
) -> Dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": base64_encode_str(text=aws_access_key),
        "AWS_SECRET_ACCESS_KEY": base64_encode_str(text=aws_secret_access_key),
        "AWS_S3_BUCKET": base64_encode_str(text=aws_s3_bucket),
        "AWS_S3_ENDPOINT": base64_encode_str(text=aws_s3_endpoint),
    }
