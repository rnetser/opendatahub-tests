import os
from typing import Optional, Tuple

import pytest
from pytest_testconfig import config as py_config


@pytest.fixture(scope="session")
def aws_access_key() -> Optional[str]:
    access_key = py_config.get("aws_access_key", os.environ.get("AWS_ACCESS_KEY_ID"))
    if not access_key:
        raise ValueError("AWS access key is not set")

    return access_key


@pytest.fixture(scope="session")
def aws_secret_access_key() -> Optional[str]:
    secret_access_key = py_config.get("aws_secret_key", os.environ.get("AWS_SECRET_ACCESS_KEY"))
    if not secret_access_key:
        raise ValueError("AWS secret key is not set")

    return secret_access_key


@pytest.fixture(scope="session")
def valid_aws_config(aws_access_key: str, aws_secret_access_key: str) -> Tuple[str, str]:
    return aws_access_key, aws_secret_access_key
