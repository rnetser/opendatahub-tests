import shlex

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.service_mesh_member import ServiceMeshMember
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.authorino import Authorino
from ocp_utilities.infra import get_pods_by_name_prefix
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.authentication.constants import (
    CAIKIT_STR,
    CAIKIT_TGIS_RUNTIME_STR,
    CAIKIT_TGIS_SERVING_TEMPLATE_STR,
    GRPC_STR,
    HTTP_STR,
    SERVERLESS_STR,
)
from tests.model_serving.model_server.authentication.utils import get_s3_secret_dict
from tests.model_serving.model_server.utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def skip_if_no_authorino_operator(admin_client: DynamicClient):
    if not Authorino(
        client=admin_client,
        name="authorino",
        namespace="redhat-ods-applications-auth-provider",
    ).exists:
        pytest.skip("Authorino operator is missing from the cluster")


@pytest.fixture(scope="class")
def s3_models_storage_uri(request) -> str:
    return f"s3://{py_config['model_s3_bucket_name']}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key: str,
    aws_secret_access_key: str,
) -> Secret:
    with Secret(
        client=admin_client,
        namespace=model_namespace.name,
        name="models-bucket-secret",
        data_dict=get_s3_secret_dict(aws_access_key=aws_access_key, aws_secret_access_key=aws_secret_access_key),
    ) as secret:
        yield secret


# HTTP model serving
@pytest.fixture(scope="class")
def http_model_service_account(admin_client: DynamicClient, endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=endpoint_s3_secret.namespace,
        name=f"{HTTP_STR}-models-bucket-sa",
        secrets=[{"name": endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def http_s3_serving_runtime(
    admin_client: DynamicClient,
    service_mesh_member: ServiceMeshMember,
    model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{HTTP_STR}-{CAIKIT_TGIS_RUNTIME_STR}",
        namespace=model_namespace.name,
        template_name=CAIKIT_TGIS_SERVING_TEMPLATE_STR,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def http_s3_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    http_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{HTTP_STR}-{CAIKIT_STR}",
        namespace=model_namespace.name,
        runtime=http_s3_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=http_s3_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=SERVERLESS_STR,
        model_service_account=http_model_service_account.name,
        enable_auth=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_view_role_binding(admin_client: DynamicClient, http_model_service_account: ServiceAccount) -> RoleBinding:
    with RoleBinding(
        client=admin_client,
        namespace=http_model_service_account.namespace,
        name=f"{HTTP_STR}-{http_model_service_account.name}-view",
        role_ref_name="view",
        role_ref_kind="ClusterRole",
        subjects_kind=http_model_service_account.kind,
        subjects_name=http_model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def http_inference_token(http_model_service_account: ServiceAccount, http_view_role_binding: RoleBinding) -> str:
    return run_command(
        command=shlex.split(
            f"oc create token -n {http_model_service_account.namespace} {http_model_service_account.name}"
        )
    )[1].strip()


@pytest.fixture()
def patched_remove_authentication_isvc(
    admin_client: DynamicClient, http_s3_inference_service: InferenceService
) -> InferenceService:
    with ResourceEditor(
        patches={
            http_s3_inference_service: {
                "metadata": {
                    "annotations": {"security.opendatahub.io/enable-auth": "false"},
                }
            }
        }
    ):
        predictor_pod = get_pods_by_name_prefix(
            client=admin_client,
            pod_prefix=f"{http_s3_inference_service.name}-predictor",
            namespace=http_s3_inference_service.namespace,
        )[0]
        predictor_pod.wait_deleted()

        yield http_s3_inference_service


# HTTP model serving
@pytest.fixture(scope="class")
def grpc_model_service_account(admin_client: DynamicClient, endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=endpoint_s3_secret.namespace,
        name=f"{GRPC_STR}-models-bucket-sa",
        secrets=[{"name": endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def grpc_s3_serving_runtime(
    admin_client: DynamicClient,
    service_mesh_member: ServiceMeshMember,
    model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{GRPC_STR}-{CAIKIT_TGIS_RUNTIME_STR}",
        namespace=model_namespace.name,
        template_name=CAIKIT_TGIS_SERVING_TEMPLATE_STR,
        multi_model=False,
        enable_http=False,
        enable_grpc=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def grpc_s3_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    grpc_s3_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    grpc_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{GRPC_STR}-{CAIKIT_STR}",
        namespace=model_namespace.name,
        runtime=grpc_s3_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=grpc_s3_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=SERVERLESS_STR,
        model_service_account=grpc_model_service_account.name,
        enable_auth=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def grpc_view_role_binding(admin_client: DynamicClient, grpc_model_service_account: ServiceAccount) -> RoleBinding:
    with RoleBinding(
        client=admin_client,
        namespace=grpc_model_service_account.namespace,
        name=f"{GRPC_STR}-{grpc_model_service_account.name}-view",
        role_ref_name="view",
        role_ref_kind="ClusterRole",
        subjects_kind=grpc_model_service_account.kind,
        subjects_name=grpc_model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def grpc_inference_token(grpc_model_service_account: ServiceAccount, grpc_view_role_binding: RoleBinding) -> str:
    return run_command(
        command=shlex.split(
            f"oc create token -n {grpc_model_service_account.namespace} {grpc_model_service_account.name}"
        )
    )[1].strip()
