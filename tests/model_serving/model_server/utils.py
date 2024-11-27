from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.private_endpoint.utils import InvalidStorageArgument


@contextmanager
def create_isvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    deployment_mode: str,
    model_format: str,
    runtime: str,
    storage_uri: Optional[str] = None,
    storage_key: Optional[str] = None,
    storage_path: Optional[str] = None,
    wait: bool = True,
    enable_auth: bool = False,
    external_route: bool = False,
    model_service_account: Optional[str] = "",
    min_replicas: Optional[int] = None,
) -> Generator[InferenceService, Any, Any]:
    predictor_dict: Dict[str, Any] = {
        "minReplicas": min_replicas,
        "model": {
            "modelFormat": {"name": model_format},
            "version": "1",
            "runtime": runtime,
        },
    }

    _check_storage_arguments(storage_uri, storage_key, storage_path)
    if storage_uri:
        predictor_dict["model"]["storageUri"] = storage_uri
    elif storage_key:
        predictor_dict["model"]["storage"] = {"key": storage_key, "path": storage_path}
    if model_service_account:
        predictor_dict["serviceAccountName"] = model_service_account

    if min_replicas:
        predictor_dict["minReplicas"] = min_replicas

    annotations = {
        "serving.knative.openshift.io/enablePassthrough": "true",
        "sidecar.istio.io/inject": "true",
        "sidecar.istio.io/rewriteAppHTTPProbers": "true",
        "serving.kserve.io/deploymentMode": deployment_mode,
    }

    if enable_auth:
        annotations["security.opendatahub.io/enable-auth"] = "true"

    with InferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations=annotations,
        predictor=predictor_dict,
    ) as inference_service:
        if wait:
            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=10 * 60,
            )
        yield inference_service


def _check_storage_arguments(
    storage_uri: Optional[str],
    storage_key: Optional[str],
    storage_path: Optional[str],
) -> None:
    if (storage_uri and storage_path) or (not storage_uri and not storage_key) or (storage_key and not storage_path):
        raise InvalidStorageArgument(storage_uri, storage_key, storage_path)
