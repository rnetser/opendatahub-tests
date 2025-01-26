from typing import Any, Generator

import pytest
from ocp_resources.data_science_cluster import DataScienceCluster

from tests.model_serving.model_server.utils import enable_model_server_components_in_dsc
from utilities.constants import DscComponents


@pytest.fixture(scope="class")
def managed_modelmesh_kserve_in_dsc(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    with enable_model_server_components_in_dsc(
        dsc=dsc_resource, components=[DscComponents.MODELMESHSERVING, DscComponents.KSERVE], wait_for_status_ready=False
    ):
        yield dsc_resource
