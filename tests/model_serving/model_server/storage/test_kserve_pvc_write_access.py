import shlex
from ocp_resources.pod import ExecOnPodError
import pytest
from ocp_utilities.infra import get_pod_by_name_prefix
from typing import List


pytestmark = pytest.mark.usefixtures("valid_aws_config")


POD_SPLIT_COMMAND: List[str] = shlex.split("touch /mnt/models/test")
KSERVE_CONTAINER_NAME: str = "kserve-container"


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, serving_runtime, inference_service",
    [
        pytest.param(
            {"name": "pvc-write-access"},
            {"model-dir": "test-dir"},
            {
                "name": "ovms-runtime",
                "model-name": "onnx",
                "model-version": "1",
                "multi-model": False,
            },
            {"name": "onnx"},
        )
    ],
    indirect=True,
)
class TestKservePVCWriteAccess:
    def test_isvc_read_only_annotation_not_set(self, inference_service):
        assert inference_service.instance.metadata.annotations.get(
            "storage.kserve.io/readonly"
        ), "Read only annotation is set"

    def test_isvc_read_only_annotation_default_value(self, predictor_pod):
        with pytest.raises(ExecOnPodError):
            predictor_pod.execute(
                container=KSERVE_CONTAINER_NAME,
                command=POD_SPLIT_COMMAND,
            )

    @pytest.mark.parametrize(
        "patched_isvc",
        [
            pytest.param(
                {"readonly": "false"},
            ),
        ],
        indirect=True,
    )
    def test_isvc_read_only_annotation_false(self, admin_client, patched_isvc):
        new_pod = get_pod_by_name_prefix(
            client=admin_client,
            pod_prefix=f"{patched_isvc.name}-predictor",
            namespace=patched_isvc.namespace,
        )
        new_pod.execute(
            container=KSERVE_CONTAINER_NAME,
            command=POD_SPLIT_COMMAND,
        )

    @pytest.mark.parametrize(
        "patched_isvc",
        [
            pytest.param(
                {"readonly": "true"},
            ),
        ],
        indirect=True,
    )
    def test_isvc_read_only_annotation_true(self, admin_client, patched_isvc):
        new_pod = get_pod_by_name_prefix(
            client=admin_client,
            pod_prefix=f"{patched_isvc.name}-predictor",
            namespace=patched_isvc.namespace,
        )
        with pytest.raises(ExecOnPodError):
            new_pod.execute(
                container=KSERVE_CONTAINER_NAME,
                command=POD_SPLIT_COMMAND,
            )
