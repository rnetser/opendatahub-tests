import pytest
from simple_logger.logger import get_logger

from utilities.constants import ComponentManagementState, DscComponents

LOGGER = get_logger(name=__name__)
TIMEOUT = 6 * 60


@pytest.mark.usefixtures("managed_modelmesh_kserve_in_dsc")
class TestKserveModelmeshCoexist:
    def test_model_mesh_state_in_dsc(self, dsc_resource):
        LOGGER.info(f"Verify {DscComponents.MODELMESHSERVING} state in DSC is {ComponentManagementState.MANAGED}")
        dsc_resource.wait_for_condition(condition="ModelMeshServingReady", status="True", timeout=TIMEOUT)

    def test_kserve_state_in_dsc(self, dsc_resource):
        LOGGER.info(f"Verify {DscComponents.KSERVE} state in DSC is {ComponentManagementState.MANAGED}")
        dsc_resource.wait_for_condition(condition="kserveReady", status="True", timeout=TIMEOUT)
