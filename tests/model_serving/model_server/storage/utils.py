import base64
import json
import shlex
from typing import Dict

from pyhelper_utils.shell import run_command


def base64_encode_str(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def run_grpc_command(url: str, query: Dict[str, str]) -> str:
    """
    Run grpc command
    Args:
        url: grpc url
        query: model query
    Returns:
        response: grpc response
    """
    cmd = (
        f"grpcurl -insecure -d '{json.dumps(query)}' -H \"mm-model-id: flan-t5-small-caikit\" "
        f"{url}:443 caikit.runtime.Nlp.NlpService/TextGenerationTaskPredict"
    )

    return run_command(command=shlex.split(cmd))[1]
