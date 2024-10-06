import shlex
from ocp_resources.pod import ExecOnPodError

import pytest


pytestmark = pytest.mark.usefixtures("valid_aws_config")


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
    def test_isvc_read_only_annotation_default_value(self, predictor_pod):
        with pytest.raises(ExecOnPodError):
            predictor_pod.execute(
                container="kserve-container",
                command=shlex.split("touch /mnt/models/test"),
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
    def test_isvc_read_only_annotation_false(self, patched_isvc, predictor_pod):
        with pytest.raises(ExecOnPodError):
            predictor_pod.execute(
                container="kserve-container",
                command=shlex.split("touch /mnt/models/test"),
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
    def test_isvc_read_only_annotation_true(self, patched_isvc, predictor_pod):
        predictor_pod.execute(
            container="kserve-container",
            command=shlex.split("touch /mnt/models/test"),
        )

    def test_isvc_modified_read_only_annotation(self, inference_service):
        pass
