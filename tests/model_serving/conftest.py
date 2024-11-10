from typing import Optional, Tuple

import pytest


@pytest.fixture(scope="session")
def aws_access_key_id(pytestconfig) -> Optional[str]:
    access_key = pytestconfig.option.aws_access_key_id
    if not access_key:
        raise ValueError(
            "AWS access key id is not set. "
            "Either pass with `--aws-access-key-id` or set `AWS_ACCESS_KEY_ID` environment variable"
        )

    return access_key


@pytest.fixture(scope="session")
def aws_secret_access_key(pytestconfig) -> Optional[str]:
    secret_access_key = pytestconfig.option.aws_secret_access_key
    if not secret_access_key:
        raise ValueError(
            "AWS secret access key is not set. "
            "Either pass with `--aws-secret-access-key` or set `AWS_SECRET_ACCESS_KEY` environment variable"
        )

    return secret_access_key


@pytest.fixture(scope="session")
def valid_aws_config(aws_access_key_id: str, aws_secret_access_key: str) -> Tuple[str, str]:
    return aws_access_key_id, aws_secret_access_key
