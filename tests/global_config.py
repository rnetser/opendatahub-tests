import os
from typing import Optional, Any, Dict

global config
config: Dict[Any, Any] = {}

# AWS
aws_secret_access_key: Optional[str] = os.environ.get("AWS_SECRET_ACCESS_KEY", "aws_secret_key")
aws_access_key_id: Optional[str] = os.environ.get("AWS_ACCESS_KEY_ID", "aws_access_key")

# S3 Buckets
ci_s3_bucket_name: str = "ci-bucket"

for _dir in dir():
    val = locals()[_dir]
    if type(val) not in [bool, list, dict, str, int]:
        continue

    if _dir in ["encoding", "py_file"]:
        continue

    config[_dir] = locals()[_dir]  # type: ignore[unused-ignore] # noqa: F821
