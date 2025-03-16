from contextlib import contextmanager
from typing import Any, Generator

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout
from utilities.infra import get_pods_by_isvc_label, wait_for_inference_deployment_replicas

LOGGER = get_logger(name=__name__)


@contextmanager
def update_inference_service(
    client: DynamicClient, isvc: InferenceService, isvc_updated_dict: dict[str, Any]
) -> Generator[InferenceService, Any, None]:
    """
    Update InferenceService object.

    Args:
        client (DynamicClient): DynamicClient object.
        isvc (InferenceService): InferenceService object.
        isvc_updated_dict (dict[str, Any]): InferenceService object.

    """
    deployment = Deployment(
        client=client,
        name=f"{isvc.name}-predictor",
        namespace=isvc.namespace,
    )
    start_generation = deployment.instance.status.observedGeneration
    wait_for_inference_deployment_replicas(
        client=client,
        isvc=isvc,
    )

    orig_pods = get_pods_by_isvc_label(client=client, isvc=isvc)

    with ResourceEditor(patches={isvc: isvc_updated_dict}):
        # Wait for new deployment generation and new pod to be created after ISVC update
        wait_for_new_deployment_generation(deployment=deployment, start_generation=start_generation)
        wait_for_inference_deployment_replicas(
            client=client,
            isvc=isvc,
        )
        wait_for_new_running_inference_pods(isvc=isvc, orig_pods=orig_pods)

        yield isvc


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
    unset_pods = []
    checked_env_vars_names = [env_var["name"] for env_var in env_vars]

    pods = get_pods_by_isvc_label(client=isvc.client, isvc=isvc)

    for pod in pods:
        pod_env_vars_names = [env_var.name for env_var in pod.instance.spec.containers[0].get("env", [])]
        envs_in_pod = [env_var in pod_env_vars_names for env_var in checked_env_vars_names]

        if vars_exist:
            if not all(envs_in_pod):
                unset_pods.append(pod.name)

        else:
            if all(envs_in_pod):
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


def wait_for_new_running_inference_pods(isvc: InferenceService, orig_pods: list[Pod]) -> None:
    """
    Wait for the inference pod to be replaced.

    Args:
        isvc (InferenceService): InferenceService object.
        orig_pods (list): List of Pod objects.

    Raises:
        TimeoutError: If the pods are not replaced.

    """
    LOGGER.info("Waiting for pods to be replaced")
    oring_pods_names = [pod.name for pod in orig_pods]

    try:
        for pods in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_2MIN,
            sleep=5,
            func=get_pods_by_isvc_label,
            client=isvc.client,
            isvc=isvc,
        ):
            if pods and len(pods) == len(orig_pods):
                for pod in pods:
                    if pod.name not in oring_pods_names and pod.status == pod.Status.RUNNING:
                        return

    except TimeoutError:
        LOGGER.error(f"Timeout waiting for pods {oring_pods_names} to be replaced")
        raise
