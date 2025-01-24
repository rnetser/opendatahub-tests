from typing import Any, Generator

import pytest
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.components.constants import KSERVE, MODELMESHSERVING
from utilities.constants import ComponentManagementState


@pytest.fixture(scope="class")
def kserve_management_state(dsc_resource: DataScienceCluster) -> str:
    return dsc_resource.instance.spec.components[KSERVE].managementState


@pytest.fixture(scope="class")
def modelmesh_management_state(dsc_resource: DataScienceCluster) -> str:
    return dsc_resource.instance.spec.components[MODELMESHSERVING].managementState


@pytest.fixture(scope="class")
def managed_modelmesh_kserve_in_dsc(
    dsc_resource: DataScienceCluster,
    modelmesh_management_state: bool,
    kserve_management_state: str,
) -> Generator[DataScienceCluster, Any, Any]:
    dsc_dict = {}

    if modelmesh_management_state == ComponentManagementState.REMOVED:
        dsc_dict.setdefault("spec", {}).setdefault("components", {})[MODELMESHSERVING] = {
            "managementState": ComponentManagementState.MANAGED
        }

    if kserve_management_state == ComponentManagementState.REMOVED:
        dsc_dict.setdefault("spec", {}).setdefault("components", {})[KSERVE] = {
            "managementState": ComponentManagementState.MANAGED
        }

    if dsc_dict:
        with ResourceEditor(patches={dsc_resource: dsc_dict}):
            yield dsc_resource

    else:
        yield dsc_resource
