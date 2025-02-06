from __future__ import annotations

import json
import os
import shlex
from contextlib import contextmanager
from functools import cache
from typing import Any, Generator, Optional, Set

import kubernetes
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.catalog_source import CatalogSource
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.project_request import ProjectRequest
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

import utilities.general
from utilities.constants import Labels
from utilities.general import create_isvc_label_selector_str

LOGGER = get_logger(name=__name__)
TIMEOUT_2MIN = 2 * 60


@contextmanager
def create_ns(
    name: str,
    admin_client: Optional[DynamicClient] = None,
    unprivileged_client: Optional[DynamicClient] = None,
    teardown: bool = True,
    delete_timeout: int = 4 * 60,
    labels: Optional[dict[str, str]] = None,
) -> Generator[Namespace | Project, Any, Any]:
    """
    Create namespace with admin or unprivileged client.

    Args:
        name (str): namespace name.
        admin_client (DynamicClient): admin client.
        unprivileged_client (UnprivilegedClient): unprivileged client.
        teardown (bool): should run resource teardown
        delete_timeout (int): delete timeout.
        labels (dict[str, str]): labels dict to set for namespace

    Yields:
        Namespace | Project: namespace or project

    """
    if unprivileged_client:
        with ProjectRequest(name=name, client=unprivileged_client, teardown=teardown):
            project = Project(
                name=name,
                client=unprivileged_client,
                teardown=teardown,
                delete_timeout=delete_timeout,
            )
            project.wait_for_status(status=project.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield project

    else:
        with Namespace(
            client=admin_client,
            name=name,
            label=labels,
            teardown=teardown,
            delete_timeout=delete_timeout,
        ) as ns:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield ns


def wait_for_inference_deployment_replicas(
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str | None,
    expected_num_deployments: int = 1,
) -> list[Deployment]:
    """
    Wait for inference deployment replicas to complete.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object
        runtime_name (str): ServingRuntime name.
        expected_num_deployments (int): Expected number of deployments per InferenceService.

    Returns:
        list[Deployment]: List of Deployment objects for InferenceService.

    """
    ns = isvc.namespace
    label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="deployment", runtime_name=runtime_name)

    deployments = list(
        Deployment.get(
            label_selector=label_selector,
            client=client,
            namespace=isvc.namespace,
        )
    )

    LOGGER.info("Waiting for inference deployment replicas to complete")
    if len(deployments) == expected_num_deployments:
        for deployment in deployments:
            if deployment.exists:
                deployment.wait_for_replicas()

        return deployments

    elif len(deployments) > expected_num_deployments:
        raise ResourceNotUniqueError(f"Multiple predictor deployments found in namespace {ns}")

    else:
        raise ResourceNotFoundError(f"Predictor deployment not found in namespace {ns}")


@contextmanager
def s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, Any, Any]:
    """
    Create S3 endpoint secret.

    Args:
        admin_client (DynamicClient): Dynamic client.
        name (str): Secret name.
        namespace (str): Secret namespace name.
        aws_access_key (str): Secret access key.
        aws_secret_access_key (str): Secret access key.
        aws_s3_bucket (str): Secret s3 bucket.
        aws_s3_endpoint (str): Secret s3 endpoint.
        aws_s3_region (str): Secret s3 region.

    Yield:
        Secret: Secret object

    """
    # DO not create secret if exists in the namespace
    os.environ["REUSE_IF_RESOURCE_EXISTS"] = f"{{Secret: {{{name}: {namespace}}}}}"

    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={"opendatahub.io/connection-type": "s3"},
        # the labels are needed to set the secret as data connection by odh-model-controller
        label={"opendatahub.io/managed": "true", Labels.OpenDataHub.DASHBOARD: "true"},
        data_dict=utilities.general.get_s3_secret_dict(
            aws_access_key=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_bucket=aws_s3_bucket,
            aws_s3_endpoint=aws_s3_endpoint,
            aws_s3_region=aws_s3_region,
        ),
        wait_for_resource=True,
    ) as secret:
        yield secret


@contextmanager
def create_isvc_view_role(
    client: DynamicClient,
    isvc: InferenceService,
    name: str,
    resource_names: Optional[list[str]] = None,
) -> Generator[Role, Any, Any]:
    """
    Create a view role for an InferenceService.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object.
        name (str): Role name.
        resource_names (list[str]): Resource names to be attached to role.

    Yields:
        Role: Role object.

    """
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


def login_with_user_password(api_address: str, user: str, password: str | None = None) -> bool:
    """
    Log in to an OpenShift cluster using a username and password.

    Args:
        api_address (str): The API address of the OpenShift cluster.
        user (str): Cluster's username
        password (str, optional): Cluster's password

    Returns:
        bool: True if login is successful otherwise False.
    """
    login_command: str = f"oc login  --insecure-skip-tls-verify=true {api_address} -u {user}"
    if password:
        login_command += f" -p '{password}'"

    _, out, _ = run_command(command=shlex.split(login_command), hide_log_command=True)

    return "Login successful" in out


@cache
def is_self_managed_operator(client: DynamicClient) -> bool:
    """
    Check if the operator is self-managed.
    """
    if py_config["distribution"] == "upstream":
        return True

    if CatalogSource(
        client=client,
        name="addon-managed-odh-catalog",
        namespace=py_config["applications_namespace"],
    ).exists:
        return False

    return True


@cache
def is_managed_cluster(client: DynamicClient) -> bool:
    """
    Check if the cluster is managed.
    """
    infra = Infrastructure(client=client, name="cluster")

    if not infra.exists:
        LOGGER.warning(f"Infrastructure {infra.name} resource does not exist in the cluster")
        return False

    platform_statuses = infra.instance.status.platformStatus

    for entries in platform_statuses.values():
        if isinstance(entries, kubernetes.dynamic.resource.ResourceField):
            if tags := entries.resourceTags:
                return next(b["value"] == "true" for b in tags if b["key"] == "red-hat-managed")

    return False


def get_services_by_isvc_label(
    client: DynamicClient, isvc: InferenceService, runtime_name: str | None = None
) -> list[Service]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService): InferenceService object.
        runtime_name (str): ServingRuntime name

    Returns:
        list[Service]: A list of all matching services

    Raises:
        ResourceNotFoundError: if no services are found.
    """
    label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="service", runtime_name=runtime_name)

    if svcs := [
        svc
        for svc in Service.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
        )
    ]:
        return svcs

    raise ResourceNotFoundError(f"{isvc.name} has no services")


def get_pods_by_isvc_label(client: DynamicClient, isvc: InferenceService, runtime_name: str | None = None) -> list[Pod]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService):InferenceService object.
        runtime_name (str): ServingRuntime name

    Returns:
        list[Pod]: A list of all matching pods

    Raises:
        ResourceNotFoundError: if no pods are found.
    """
    label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="pod", runtime_name=runtime_name)

    if pods := [
        pod
        for pod in Pod.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
        )
    ]:
        return pods

    raise ResourceNotFoundError(f"{isvc.name} has no pods")


def get_openshift_token() -> str:
    """
    Get the OpenShift token.

    Returns:
        str: The OpenShift token.

    """
    return run_command(command=shlex.split("oc whoami -t"))[1].strip()


def get_kserve_storage_initialize_image(client: DynamicClient) -> str:
    """
    Get the image used to storage-initializer.

    Args:
        client (DynamicClient): DynamicClient client.

    Returns:
        str: The image used to storage-initializer.

    Raises:
        ResourceNotFoundError: if the config map does not exist.

    """
    kserve_cm = ConfigMap(
        client=client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
    )

    if not kserve_cm.exists:
        raise ResourceNotFoundError(f"{kserve_cm.name} config map does not exist")

    return json.loads(kserve_cm.instance.data.storageInitializer)["image"]


def get_inference_serving_runtime(isvc: InferenceService) -> ServingRuntime:
    """
    Get the serving runtime for the inference service.

    Args:
        isvc (InferenceService):InferenceService object.

    Returns:
        ServingRuntime: ServingRuntime object.

    Raises:
        ResourceNotFoundError: if the serving runtime does not exist.

    """
    runtime = ServingRuntime(
        client=isvc.client,
        namespace=isvc.namespace,
        name=isvc.instance.spec.predictor.model.runtime,
    )

    if runtime.exists:
        return runtime

    raise ResourceNotFoundError(f"{isvc.name} runtime {runtime.name} does not exist")


def get_model_mesh_route(client: DynamicClient, isvc: InferenceService) -> Route:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService):InferenceService object.

    Returns:
        Route: inference service route

    Raises:
        ResourceNotFoundError: if route was found.
    """
    if routes := [
        route
        for route in Route.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=f"inferenceservice-name={isvc.name}",
        )
    ]:
        return routes[0]

    raise ResourceNotFoundError(f"{isvc.name} has no routes")


def create_inference_token(model_service_account: ServiceAccount) -> str:
    """
    Generates an inference token for the given model service account.

    Args:
        model_service_account (ServiceAccount): An object containing the namespace and name
                               of the service account.

    Returns:
        str: The generated inference token.
    """
    return run_command(
        shlex.split(f"oc create token -n {model_service_account.namespace} {model_service_account.name}")
    )[1].strip()


@contextmanager
def update_configmap_data(
    client: DynamicClient, name: str, namespace: str, data: dict[str, Any]
) -> Generator[ConfigMap, Any, Any]:
    """
    Update the data of a configmap.

    Args:
        client (DynamicClient): DynamicClient client.
        name (str): Name of the configmap.
        namespace (str): Namespace of the configmap.
        data (dict[str, Any]): Data to update the configmap with.

    Yields:
        ConfigMap: The updated configmap.

    """
    config_map = ConfigMap(client=client, name=name, namespace=namespace)

    # Some CM resources may already be present as they are usually created when doing exploratory testing
    if config_map.exists:
        with ResourceEditor(patches={config_map: {"data": data}}):
            yield config_map

    else:
        config_map.data = data
        with config_map as cm:
            yield cm


def check_pod_status_in_time(pod: Pod, status: Set[str], duration: int = TIMEOUT_2MIN, wait: int = 1) -> None:
    """
    Checks if a pod status is maintained for a given duration. If not, an AssertionError is raised.

    Args:
        pod (Pod): The pod to check
        status (Set[Pod.Status]): Expected pod status(es)
        duration (int): Maximum time to check for in seconds
        wait (int): Time to wait between checks in seconds

    Raises:
        AssertionError: If pod status is not in the expected set
    """
    LOGGER.info(f"Checking pod status for {pod.name} to be {status} for {duration} seconds")

    sampler = TimeoutSampler(
        wait_timeout=duration,
        sleep=wait,
        func=lambda: pod.instance,
    )

    try:
        for sample in sampler:
            if sample:
                if sample.status.phase not in status:
                    raise AssertionError(f"Pod status is not the expected: {pod.status}")

    except TimeoutExpiredError:
        LOGGER.info(f"Pod status is {pod.status} as expected")
