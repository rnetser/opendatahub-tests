import pytest

from utilities.constants import (
    KServeDeploymentType,
    ModelStoragePath,
    RuntimeTemplates,
)

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.sanity
@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-isvc-update"},
            {
                "name": "raw-isvc-envs-update",
                "template-name": RuntimeTemplates.CAIKIT_STANDALONE_SERVING,
                "multi-model": False,
                "enable-http": True,
            },
            {
                "name": "isvc-envs-update",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": ModelStoragePath.EMBEDDING_MODEL,
            },
        )
    ],
    indirect=True,
)
class TestISVCEnvVarsUpdates:
    @pytest.mark.parametrize(
        "patched_isvc_env_vars",
        [
            pytest.param({"action": "add-envs"}),
        ],
        indirect=True,
    )
    def test_add_isvc_env_vars(self, patched_isvc_env_vars):
        """Test adding environment variables to the inference service"""
        pass

    @pytest.mark.parametrize(
        "patched_isvc_env_vars",
        [
            pytest.param({"action": "remove-envs"}),
        ],
        indirect=True,
    )
    def test_remove_isvc_env_vars(self, patched_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        pass
