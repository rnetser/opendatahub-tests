from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.utils import verify_inference_response


def run_inference_multiple_times(
    isvc: InferenceService,
    runtime: str,
    inference_type: str,
    protocol: str,
    model_name: str,
    iterations: int,
) -> None:
    for iteration in range(iterations):
        verify_inference_response(
            inference_service=isvc,
            runtime=runtime,
            inference_type=inference_type,
            protocol=protocol,
            model_name=model_name,
            use_default_query=True,
        )
