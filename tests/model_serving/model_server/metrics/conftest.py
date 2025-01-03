import pytest
from kubernetes.dynamic import DynamicClient
from ocp_utilities.monitoring import Prometheus

from utilities.infra import get_openshift_token


# @pytest.fixture(scope="session")
# def thanos_url(admin_client: DynamicClient) -> str:
#     for route in Route.get(dyn_client=admin_client, name="thanos-querier", namespace="openshift-monitoring"):
#         if route.exists:
#             return f"https://{route.instance.spec.host}"
#
#     raise ResourceNotFoundError("Cannot find thanos-querier route")


@pytest.fixture(scope="session")
def prometheus(admin_client: DynamicClient) -> Prometheus:
    return Prometheus(
        client=admin_client,
        resource_name="thanos-querier",
        verify_ssl=False,
        bearer_token=get_openshift_token(),
    )
