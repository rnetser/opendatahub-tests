import pytest
from simple_logger.logger import get_logger

from tests.model_serving.model_server.components.constants import KSERVE
from utilities.constants import ComponentManagementState

LOGGER = get_logger(name=__name__)
TIMEOUT = 6 * 60


@pytest.mark.usefixtures("managed_modelmesh_kserve_in_dsc")
class TestKserveModelmeshCoexist:
    def test_model_mesh_state_in_dsc(self, dsc_resource):
        LOGGER.info(f"Verify model mesh state in DSC is {ComponentManagementState.MANAGED}")
        dsc_resource.wait_for_condition(condition="ModelMeshServingReady", status="True", timeout=TIMEOUT)

    def test_kserve_state_in_dsc(self, dsc_resource):
        LOGGER.info(f"Verify {KSERVE} state in DSC is {ComponentManagementState.MANAGED}")
        dsc_resource.wait_for_condition(condition="kserveReady", status="True", timeout=TIMEOUT)
