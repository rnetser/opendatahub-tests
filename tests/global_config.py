import os
from typing import Optional, Any, Dict

global config
config: Dict[Any, Any] = {}

no_unprivileged_client: bool = False
# AWS
aws_secret_access_key: Optional[str] = os.environ.get("AWS_SECRET_ACCESS_KEY", "aws_secret_key")
aws_access_key_id: Optional[str] = os.environ.get("AWS_ACCESS_KEY_ID", "aws_access_key")

# S3
ci_s3_bucket_name: str = "ci-bucket"
model_s3_bucket_name: str = "s3-bucket"
model_s3_bucket_region: str = "us-east-1"
model_s3_endpoint: str = f"https://{model_s3_bucket_region}.amazonaws.com/"

for _dir in dir():
    val = locals()[_dir]
    if type(val) not in [bool, list, dict, str, int]:
        continue

    if _dir in ["encoding", "py_file"]:
        continue

    config[_dir] = locals()[_dir]  # type: ignore[unused-ignore] # noqa: F821
