from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.inference_service_configuration.constants import ISVC_ENV_VARS
from tests.model_serving.model_server.inference_service_configuration.utils import (
    wait_for_new_deployment_generation,
    wait_for_new_running_inference_pod,
)
from utilities.infra import get_pods_by_isvc_label


@pytest.fixture(scope="class")
def removed_isvc_env_vars(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    isvc_predictor_spec_model_env = ovms_kserve_inference_service.instance.spec.predictor.model.get("env", [])
    isvc_predictor_spec_model_env = [
        env_var for env_var in isvc_predictor_spec_model_env if env_var not in ISVC_ENV_VARS
    ]

    deployment = Deployment(
        client=admin_client,
        name=f"{ovms_kserve_inference_service.name}-predictor",
        namespace=ovms_kserve_inference_service.namespace,
    )
    start_generation = deployment.instance.status.observedGeneration

    orig_pod = get_pods_by_isvc_label(client=admin_client, isvc=ovms_kserve_inference_service)[0]

    with ResourceEditor(
        patches={
            ovms_kserve_inference_service: {"spec": {"predictor": {"model": {"env": isvc_predictor_spec_model_env}}}}
        }
    ):
        # Wait for new deployment generation and new pod to be created after ISVC update
        wait_for_new_deployment_generation(deployment=deployment, start_generation=start_generation)
        wait_for_new_running_inference_pod(isvc=ovms_kserve_inference_service, orig_pod=orig_pod)

        yield ovms_kserve_inference_service


@pytest.fixture(scope="class")
def patched_isvc_replicas(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    deployment = Deployment(
        client=admin_client,
        name=f"{ovms_kserve_inference_service.name}-predictor",
        namespace=ovms_kserve_inference_service.namespace,
    )
    start_generation = deployment.instance.status.observedGeneration
    orig_pod = get_pods_by_isvc_label(client=admin_client, isvc=ovms_kserve_inference_service)[0]

    with ResourceEditor(
        patches={
            ovms_kserve_inference_service: {
                "spec": {
                    "predictor": {
                        "maxReplicas": request.param["min-replicas"],
                        "minReplicas": request.param["min-replicas"],
                    }
                }
            }
        }
    ):
        # Wait for new deployment generation and new pod to be created after ISVC update
        wait_for_new_deployment_generation(deployment=deployment, start_generation=start_generation)
        wait_for_new_running_inference_pod(isvc=ovms_kserve_inference_service, orig_pod=orig_pod)

        yield ovms_kserve_inference_service
