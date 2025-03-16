from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.inference_service_configuration.constants import (
    ISVC_ENV_VARS,
)
from tests.model_serving.model_server.inference_service_configuration.utils import (
    update_inference_service,
)


@pytest.fixture(scope="class")
def removed_isvc_env_vars(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    isvc_predictor_spec_model_env = ovms_kserve_inference_service.instance.spec.predictor.model.get("env", [])
    isvc_predictor_spec_model_env = [
        env_var for env_var in isvc_predictor_spec_model_env if env_var.to_dict() not in ISVC_ENV_VARS
    ]

    with update_inference_service(
        client=admin_client,
        isvc=ovms_kserve_inference_service,
        isvc_updated_dict={"spec": {"predictor": {"model": {"env": isvc_predictor_spec_model_env}}}},
    ):
        yield ovms_kserve_inference_service
