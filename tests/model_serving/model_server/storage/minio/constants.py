from utilities.constants import MinIo, ModelAndFormat

MINIO_DATA_CONNECTION_CONFIG = {"bucket": MinIo.Buckets.EXAMPLE_MODELS}
MINIO_RUNTIME_CONFIG = {
    "runtime-name": f"{MinIo.Metadata.NAME}-ovms",
    "supported-model-formats": [{"name": ModelAndFormat.OPENVINO_IR, "version": "1"}],
    "runtime_image": "quay.io/jooholee/model-minio@sha256:b50aa0fbfea740debb314ece8e925b3e8e761979f345b6cd12a6833efd04e2c2",  # noqa: E501
}
MINIO_INFERENCE_CONFIG = {
    "name": "loan-model",
    "model-format": ModelAndFormat.OPENVINO_IR,
    "model-version": "1",
    "model-dir": "kserve/openvino-age-gender-recognition",
}
INFERENCE_TYPE = "age-gender-recognition"
