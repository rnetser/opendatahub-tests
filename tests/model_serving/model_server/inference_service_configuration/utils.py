from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout
from utilities.infra import get_pods_by_isvc_label


LOGGER = get_logger(name=__name__)


def verify_env_vars_in_isvc_pods(isvc: InferenceService, env_vars: list[dict[str, str]], vars_exist: bool) -> None:
    """
    Verify that the environment variables in the InferenceService pods match the expected values.

    Args:
        isvc (InferenceService): InferenceService object.
        env_vars (list[dict[str, str]]): List of environment variables to verify.
        vars_exist (bool): Whether the environment variables should exist in the pod.

    Raises:
        ValueError: If the environment variables do not match the expected values.
    """
    pods = get_pods_by_isvc_label(client=isvc.client, isvc=isvc)

    unset_pods = []
    for pod in pods:
        pod_env_vars = [env_var.name for env_var in pod.instance.spec.containers[0].env]
        expected_env_vars = [env_var["name"] for env_var in env_vars]

        if vars_exist:
            if not all([env_var in pod_env_vars for env_var in expected_env_vars]):
                unset_pods.append(pod.name)
        else:
            if all([env_var not in pod_env_vars for env_var in expected_env_vars]):
                unset_pods.append(pod.name)

    if unset_pods:
        raise ValueError(
            f"The environment variables are {'not' if vars_exist else ''} set in the following pods: {unset_pods}"
        )


def wait_for_new_deployment_generation(deployment: Deployment, start_generation: int) -> None:
    """
    Wait for the deployment generation to be updated.

    Args:
        deployment (Deployment): Dynamic client.
        start_generation (int): The start generation of the deployment.

    Raises:
        TimeoutError: If the deployment generation is not updated.

    """
    LOGGER.info(f"Waiting for deployment generation to be updated, original generation: {start_generation}")
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


def wait_for_new_running_inference_pod(isvc: InferenceService, orig_pod: Pod) -> None:
    """
    Wait for the inference pod to be replaced.

    Args:
        isvc (InferenceService): InferenceService object.
        orig_pod (Pod): Pod object.

    Raises:
        TimeoutError: If the pod is not replaced.

    """
    LOGGER.info(f"Waiting for pod {orig_pod.name} to be replaced")
    try:
        for pods in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_2MIN,
            sleep=5,
            func=get_pods_by_isvc_label,
            client=isvc.client,
            isvc=isvc,
        ):
            if pods:
                for pod in pods:
                    if pod.name != orig_pod.name and pod.status == pod.Status.RUNNING:
                        return

    except TimeoutError:
        LOGGER.error(f"Timeout waiting for pod {orig_pod.name} to be replaced")
        raise
