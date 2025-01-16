import shlex
from typing import Any, Generator
from urllib.parse import urlparse
from contextlib import contextmanager

from ocp_resources.pod import Pod
from kubernetes.dynamic.client import DynamicClient
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from utilities.constants import Protocols
from utilities.exceptions import ProtocolNotSupported

LOGGER = get_logger(name=__name__)


def curl_from_pod(
    isvc: InferenceService,
    pod: Pod,
    endpoint: str,
    protocol: str = Protocols.HTTP,
) -> str:
    if protocol not in (Protocols.HTTPS, Protocols.HTTP):
        raise ProtocolNotSupported(protocol)
    host = isvc.instance.status.address.url
    if protocol == "http":
        parsed = urlparse(host)
        host = parsed._replace(scheme="http").geturl()
    return pod.execute(command=shlex.split(f"curl -k {host}/{endpoint}"), ignore_rc=True)


@contextmanager
def create_sidecar_pod(
    admin_client: DynamicClient,
    namespace: str,
    use_istio: bool,
    pod_name: str,
) -> Generator[Pod, Any, Any]:
    containers = [
        {
            "name": pod_name,
            "image": "registry.access.redhat.com/rhel7/rhel-tools",
            "imagePullPolicy": "Always",
            "args": ["sleep", "infinity"],
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "seccompProfile": {"type": "RuntimeDefault"},
                "capabilities": {"drop": ["ALL"]},
            },
        }
    ]

    pod_kwargs = {"client": admin_client, "name": pod_name, "namespace": namespace, "containers": containers}

    if use_istio:
        pod_kwargs.update({"annotations": {"sidecar.istio.io/inject": "true"}})

    with Pod(**pod_kwargs) as pod:
        pod.wait_for_condition(condition="Ready", status="True")
        yield pod
