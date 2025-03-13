from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout
from utilities.infra import get_pods_by_isvc_label


LOGGER = get_logger(name=__name__)


def verify_env_vars_in_isvc_pod(isvc: InferenceService, env_vars: list[dict[str, str]], vars_exist: bool) -> None:
    """

    Args:
        isvc (InferenceService): InferenceService object.
        env_vars (list[dict[str, str]]): List of environment variables to verify.
        vars_exist (bool): Whether the environment variables should exist in the pod.
    Raises:
        ValueError: If the environment variables do not match the expected values.
    """
    pod = get_pods_by_isvc_label(client=isvc.client, isvc=isvc)[0]
    pod_env_vars = [env_var.name for env_var in pod.instance.spec.containers[0].env]
    expected_env_vars = [env_var["name"] for env_var in env_vars]

    if vars_exist:
        assert all(env_var in pod_env_vars for env_var in expected_env_vars)
    else:
        assert all(env_var not in pod_env_vars for env_var in expected_env_vars)


def wait_for_new_deployment_generation(client: DynamicClient, isvc: InferenceService) -> None:
    """
    Wait for the deployment generation to be updated.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object.

    Raises:
        AssertionError: If the deployment generation is not updated.

    """
    deployment = Deployment(client=client, name=f"{isvc}-predictor", namespace=isvc.namespace)

    start_generation = deployment.instance.status.observedGeneration

    try:
        for generation in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_2MIN,
            sleep=5,
            func=lambda: deployment.instance.status.observedGeneration,
        ):
            if generation and generation > start_generation:
                return

    except TimeoutError:
        LOGGER.error(f"Timeout waiting for deployment generation, original generation: {start_generation}")
        raise
