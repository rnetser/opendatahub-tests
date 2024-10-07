import io
import yaml
from typing import Any, Dict
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template


class ServingRuntimeFromTemplate(ServingRuntime):
    def __init__(self, client: DynamicClient, name: str, namespace: str, template_name: str):
        self.client = client
        self.name = name
        self.namespace = namespace
        self.template_name = template_name
        self.yaml_file = self.get_model_from_template()

        super().__init__(client=self.client, yaml_file=self.yaml_file)

    def get_model_template(self) -> Template:
        template = Template(
            client=self.client,
            name=self.template_name,
            namespace="redhat-ods-applications",
        )
        if template.exists:
            return template

        raise ResourceNotFoundError(f"{self.template_name} template not found")

    def get_model_from_template(self) -> io.StringIO:
        template = self.get_model_template()
        model: Dict[str, Any] = template.instance.objects[0].to_dict()
        model["metadata"]["name"] = self.name
        model["metadata"]["namespace"] = self.namespace

        return io.StringIO(yaml.dump(model))
