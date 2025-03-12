import pytest
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from simple_logger.logger import get_logger

from tests.model_serving.model_server.multi_node.constants import HEAD_POD_ROLE, WORKER_POD_ROLE
from tests.model_serving.model_server.multi_node.utils import (
    verify_nvidia_gpu_status,
    verify_ray_status,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols, Timeout
from utilities.infra import verify_no_failed_pods
from utilities.manifests.vllm import VLLM_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("skip_if_no_gpu_nodes", "skip_if_no_nfs_storage_class"),
]


MODEL_DIR = "granite-8b-code-base"
NAMESPACE_NAME = "gpu-multi-node"
LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace_scope_module, models_bucket_downloaded_model_data_scope_module, multi_node_inference_service",
    [
        pytest.param(
            {"name": NAMESPACE_NAME},
            {"model-dir": MODEL_DIR},
            {"name": "multi-vllm"},
        )
    ],
    indirect=True,
)
class TestBasicMultiNode:
    def test_multi_node_ray_status(self, multi_node_predictor_pods_scope_class):
        """Test multi node ray status"""
        verify_ray_status(pods=multi_node_predictor_pods_scope_class)

    def test_multi_node_nvidia_gpu_status(self, multi_node_predictor_pods_scope_class):
        """Test multi node ray status"""
        verify_nvidia_gpu_status(pod=multi_node_predictor_pods_scope_class[0])

    def test_multi_node_default_config(self, multi_node_serving_runtime, multi_node_predictor_pods_scope_class):
        """Test multi node inference service with default config"""
        runtime_worker_spec = multi_node_serving_runtime.instance.spec.workerSpec

        if runtime_worker_spec.tensorParallelSize != 1 or runtime_worker_spec.pipelineParallelSize != 2:
            pytest.fail(f"Multinode runtime default worker spec is not as expected, {runtime_worker_spec}")

    def test_multi_node_pods_distribution(self, multi_node_predictor_pods_scope_class, nvidia_gpu_nodes):
        """Verify multi node pods are distributed between cluster GPU nodes"""
        pods_nodes = {pod.node.name for pod in multi_node_predictor_pods_scope_class}
        assert len(multi_node_predictor_pods_scope_class) == len(pods_nodes), (
            "Pods are not distributed between cluster GPU nodes"
        )

        assert pods_nodes.issubset({node.name for node in nvidia_gpu_nodes}), "Pods not running on GPU nodes"

    def test_multi_node_basic_internal_inference(self, multi_node_inference_service):
        """Test multi node basic internal inference"""
        verify_inference_response(
            inference_service=multi_node_inference_service,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    def test_multi_node_basic_external_inference(self, patched_multi_node_isvc_external_route):
        """Test multi node basic external inference"""
        verify_inference_response(
            inference_service=patched_multi_node_isvc_external_route,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "model_namespace_scope_module, models_bucket_downloaded_model_data_scope_module, multi_node_inference_service",
    [
        pytest.param(
            {"name": NAMESPACE_NAME},
            {"model-dir": MODEL_DIR},
            {"name": "multi-vllm-tls"},
        )
    ],
    indirect=True,
)
class TestMultiNodeTLS:
    def test_tls_secret_exists_in_control_ns(self, multi_node_inference_service, ray_ca_tls_secret):
        """Test multi node ray ca tls secret exists in control (applications) namespace"""
        if not ray_ca_tls_secret.exists:
            raise ResourceNotFoundError(
                f"Secret {ray_ca_tls_secret.name} does not exist in {ray_ca_tls_secret.namespace} namespace"
            )

    def test_tls_secret_exists_in_inference_ns(self, ray_tls_secret):
        """Test multi node ray tls secret exists in isvc namespace"""
        if not ray_tls_secret.exists:
            raise ResourceNotFoundError(f"Secret {ray_tls_secret.name} does not exist")

    def test_cert_files_exist_in_pods(self, multi_node_predictor_pods_scope_class):
        """Test multi node cert files exist in pods"""
        missing_certs_pods = []
        for pod in multi_node_predictor_pods_scope_class:
            certs = pod.execute(command=["ls", "/etc/ray/tls"]).split()
            if "ca.crt" not in certs or "tls.pem" not in certs:
                missing_certs_pods.append(pod.name)

        assert not missing_certs_pods, f"Missing certs in pods: {missing_certs_pods}"

    @pytest.mark.parametrize(
        "deleted_multi_node_pod",
        [pytest.param({"pod-role": HEAD_POD_ROLE})],
        indirect=True,
    )
    def test_multi_node_head_pod_deleted(self, admin_client, multi_node_inference_service, deleted_multi_node_pod):
        """Test multi node when head pod is deleted"""
        verify_no_failed_pods(
            client=admin_client,
            isvc=multi_node_inference_service,
            timeout=Timeout.TIMEOUT_10MIN,
        )

    @pytest.mark.parametrize(
        "deleted_multi_node_pod",
        [pytest.param({"pod-role": WORKER_POD_ROLE})],
        indirect=True,
    )
    def test_multi_node_worker_pod_deleted(self, admin_client, multi_node_inference_service, deleted_multi_node_pod):
        """Test multi node when worker pod is deleted"""
        verify_no_failed_pods(
            client=admin_client,
            isvc=multi_node_inference_service,
            timeout=Timeout.TIMEOUT_10MIN,
        )

    @pytest.mark.dependency(name="test_ray_ca_tls_secret_reconciliation")
    def test_ray_ca_tls_secret_reconciliation(self, multi_node_inference_service, ray_ca_tls_secret):
        """Test multi node ray ca tls secret reconciliation"""
        ray_ca_tls_secret.clean_up()
        ray_ca_tls_secret.wait()

    @pytest.mark.dependency(name="test_ray_tls_secret_reconciliation")
    def test_ray_tls_secret_reconciliation(self, ray_tls_secret):
        """Test multi node ray ca tls secret reconciliation"""
        ray_tls_secret.clean_up()
        ray_tls_secret.wait()

    @pytest.mark.usefixtures("deleted_serving_runtime")
    @pytest.mark.dependency(name="test_ray_tls_deleted_on_runtime_deletion")
    def test_ray_tls_deleted_on_runtime_deletion(self, ray_tls_secret, ray_ca_tls_secret):
        """Test multi node ray tls secret deletion on runtime deletion"""
        ray_tls_secret.wait_deleted()
        assert ray_ca_tls_secret.exists

    @pytest.mark.dependency(depends=["test_ray_tls_deleted_on_runtime_deletion"])
    def test_ray_tls_created_on_runtime_creation(self, ray_tls_secret, ray_ca_tls_secret):
        """Test multi node ray tls secret creation on runtime creation"""
        ray_tls_secret.wait()

    @pytest.mark.parametrize(
        "deleted_multi_node_pod",
        [pytest.param({"pod-role": HEAD_POD_ROLE})],
        indirect=True,
    )
    @pytest.mark.dependency(depends=["test_ray_tls_secret_reconciliation"])
    def test_multi_node_inference_after_pod_deletion(
        self, admin_client, multi_node_inference_service, deleted_multi_node_pod
    ):
        """Test multi node inference after pod deletion"""
        verify_no_failed_pods(
            client=admin_client,
            isvc=multi_node_inference_service,
            timeout=Timeout.TIMEOUT_10MIN,
        )
        verify_inference_response(
            inference_service=multi_node_inference_service,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
