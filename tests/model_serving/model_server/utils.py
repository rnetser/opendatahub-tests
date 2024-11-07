from contextlib import contextmanager
from typing import Any, Dict, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService


@contextmanager
def create_isvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    deployment_mode: str,
    storage_uri: str,
    model_format: str,
    runtime: str,
    wait: bool = True,
    enable_auth: bool = False,
    external_route: bool = False,
    model_service_account: Optional[str] = "",
    min_replicas: Optional[int] = None,
) -> InferenceService:
    predictor_config: Dict[str, Any] = {
        "model": {
            "modelFormat": {"name": model_format},
            "version": "1",
            "runtime": runtime,
            "storageUri": storage_uri,
        },
    }
    if model_service_account:
        predictor_config["serviceAccountName"] = model_service_account

    if min_replicas:
        predictor_config["minReplicas"] = min_replicas

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
        predictor=predictor_config,
    ) as inference_service:
        if wait:
            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=10 * 60,
            )

        yield inference_service
