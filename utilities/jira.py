import os
import re
from functools import cache

from jira import JIRA
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion
from packaging.version import Version
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@cache
def get_jira_connection() -> JIRA:
    """
    Get Jira connection.

    Returns:
        JIRA: Jira connection.

    """
    return JIRA(
        token_auth=os.getenv("PYTEST_JIRA_TOKEN"),
        options={"server": os.getenv("PYTEST_JIRA_URL")},
    )


def is_jira_open(jira_id: str, admin_client: DynamicClient) -> bool:
    """
    Check if Jira issue is open.

    Args:
        jira_id (str): Jira issue id.
        admin_client (DynamicClient): DynamicClient object

    Returns:
        bool: True if Jira issue is open.

    """
    jira_fields = get_jira_connection().issue(id=jira_id, fields="status, fixVersions").fields

    jira_status = jira_fields.status.name.lower()

    if jira_status not in ("testing", "resolved", "closed"):
        LOGGER.info(f"Jira {jira_id}: status is {jira_status}")
        return True

    else:
        for csv in ClusterServiceVersion.get(dyn_client=admin_client, namespace=py_config["applications_namespace"]):
            if re.match("rhods|opendatahub", csv.name):
                csv_version = csv.instance.spec.version
                for fix_version in jira_fields.fixVersions:
                    if _fix_version := re.search(r"\d.\d+.\d+", fix_version.name):
                        if Version(csv_version) < Version(_fix_version.group()):
                            LOGGER.info(
                                f"Jira {jira_id}: status is {jira_status}, "
                                f"fix version is {csv_version}, operator version is {_fix_version}"
                            )
                            return True

    return False
