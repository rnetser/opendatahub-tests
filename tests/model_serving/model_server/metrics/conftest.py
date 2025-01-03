import pytest
import requests
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_utilities.monitoring import Prometheus
from simple_logger.logger import get_logger

from utilities.infra import get_openshift_token


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def prometheus(admin_client: DynamicClient) -> Prometheus:
    return Prometheus(
        client=admin_client,
        resource_name="thanos-querier",
        verify_ssl=False,
        bearer_token=get_openshift_token(),
    )


@pytest.fixture()
def deleted_metric(request: FixtureRequest, prometheus: Prometheus) -> None:
    metric = request.param["metric-name"]

    LOGGER.info(f"deleting {metric} metric")

    requests.post(
        f"{prometheus.api_url}/api/v1/admin/tsdb/delete_series?match[]={metric}",
        headers=prometheus.headers,
        verify=prometheus.verify_ssl,
    )
