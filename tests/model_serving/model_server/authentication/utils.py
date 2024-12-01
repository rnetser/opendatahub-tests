import re
from contextlib import contextmanager
from typing import Dict, List, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.role import Role
from simple_logger.logger import get_logger

from tests.model_serving.model_server.utils import b64_encoded_string
from utilities.inference_utils import LlmInference

LOGGER = get_logger(name=__name__)


def verify_inference_response(
    inference_service: InferenceService,
    runtime: str,
    inference_type: str,
    protocol: str,
    model_name: str,
    text: Optional[str] = None,
    use_default_query: bool = False,
    expected_response_text: Optional[str] = None,
    insecure: bool = True,
    token: Optional[str] = None,
    authorized_user: Optional[bool] = None,
) -> None:
    inference = LlmInference(
        inference_service=inference_service,
        runtime=runtime,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference(
        model_name=model_name,
        text=text,
        use_default_query=use_default_query,
        token=token,
        insecure=insecure,
    )

    if authorized_user is False:
        auth_header = "x-ext-auth-reason"

        if auth_reason := re.search(rf"{auth_header}: (.*)", res["output"], re.MULTILINE):
            reason = auth_reason.group(1).lower()

            if token:
                assert re.search(r"not (?:authenticated|authorized)", reason)

            else:
                assert "credential not found" in reason

        else:
            LOGGER.error(f"Auth header {auth_header} not found in response. Response: {res['output']}")
            raise

    else:
        if use_default_query:
            expected_response_text = inference.inference_config["default_query_model"]["model"].get("response_text")

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
    aws_s3_region: str,
) -> Dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(string_to_encode=aws_access_key),
        "AWS_SECRET_ACCESS_KEY": b64_encoded_string(string_to_encode=aws_secret_access_key),
        "AWS_S3_BUCKET": b64_encoded_string(string_to_encode=aws_s3_bucket),
        "AWS_S3_ENDPOINT": b64_encoded_string(string_to_encode=aws_s3_endpoint),
        "AWS_DEFAULT_REGION": b64_encoded_string(string_to_encode=aws_s3_region),
    }


@contextmanager
def create_isvc_view_role(
    client: DynamicClient,
    isvc: InferenceService,
    name: str,
    resource_names: Optional[List[str]] = None,
) -> Role:
    rules = [
        {
            "apiGroups": [isvc.api_group],
            "resources": ["inferenceservices"],
            "verbs": ["get"],
        },
    ]

    if resource_names:
        rules[0].update({"resourceNames": resource_names})

    with Role(
        client=client,
        name=name,
        namespace=isvc.namespace,
        rules=rules,
    ) as role:
        yield role
