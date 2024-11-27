from contextlib import contextmanager
from typing import Dict, Generator, Optional

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace


@contextmanager
def create_ns(
    name: str,
    admin_client: DynamicClient,
    teardown: bool = True,
    delete_timeout: int = 6 * 10,
    labels: Optional[Dict[str, str]] = None,
) -> Generator[Namespace, None, None]:
    with Namespace(
        client=admin_client,
        name=name,
        label=labels,
        teardown=teardown,
        delete_timeout=delete_timeout,
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=2 * 10)
        yield ns


def wait_for_kserve_predictor_deployment_replicas(client: DynamicClient, isvc: InferenceService) -> Deployment:
    ns = isvc.namespace

    deployments = list(
        Deployment.get(
            label_selector=f"{isvc.ApiGroup.SERVING_KSERVE_IO}/inferenceservice={isvc.name}",
            client=client,
            namespace=isvc.namespace,
        )
    )

    if len(deployments) == 1:
        deployment = deployments[0]
        if deployment.exists:
            deployment.wait_for_replicas()
            return deployment

    elif len(deployments) > 1:
        raise ResourceNotUniqueError(f"Multiple predictor deployments found in namespace {ns}")

    else:
        raise ResourceNotFoundError(f"Predictor deployment not found in namespace {ns}")
