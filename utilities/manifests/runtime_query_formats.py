from typing import Any, Dict

RUNTIME_QUERY_FORMATS: Dict[str, Any] = {
    "caikit-tgis-runtime": {
        "all-tokens": {
            "grpc": {
                "endpoint": "caikit.runtime.Nlp.NlpService/TextGenerationTaskPredict",
                "header": "mm-model-id: $model_name",
                "body": '{"text": "$query_text"}',
                "response_fields_map": {
                    "response": "",
                    "response_tokens": "generated_tokens",
                    "response_text": "generated_text",
                },
            },
            "http": {
                "endpoint": "api/v1/task/text-generation",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_text"}',
                "response_fields_map": {
                    "response": "",
                    "response_tokens": "generated_tokens",
                    "response_text": "generated_text",
                },
            },
        },
        "streaming": {
            "grpc": {
                "endpoint": "caikit.runtime.Nlp.NlpService/ServerStreamingTextGenerationTaskPredict",
                "header": "mm-model-id: $model_name",
                "body": '{"text": "$query_text"}',
                "response_fields_map": {"response": ""},
            },
            "http": {
                "endpoint": "api/v1/task/server-streaming-text-generation",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_text"}',
                "response_fields_map": {"response": ""},
            },
        },
    },
    "tgis-runtime": {
        "all-tokens": {
            "grpc": {
                "endpoint": "fmaas.GenerationService/Generate",
                "header": "mm-model-id: $model_name",
                "body": '{"requests": [{"text":"$query_text"}]}',
                "args": "proto=text-generation-inference/proto/generation.proto",
                "response_fields_map": {
                    "response": "responses",
                    "response_tokens": "generatedTokenCount",
                    "response_text": "text",
                },
            }
        },
        "streaming": {
            "grpc": {
                "endpoint": "fmaas.GenerationService/GenerateStream",
                "header": "mm-model-id: $model_name",
                "body": '{"request": [{"text":"$query_text"}]}',
                "args": "proto=text-generation-inference/proto/generation.proto",
                "response_fields_map": {"response": ""},
            }
        },
        "tokenize": {
            "grpc": {
                "endpoint": "fmaas.GenerationService/Tokenize",
                "header": "mm-model-id: $model_name",
                "body": '{"requests": [{"text":"$query_text"}], "return_tokens":"true"}',
                "args": "proto=text-generation-inference/proto/generation.proto",
                "response_fields_map": {
                    "response": "",
                    "response_tokens": "",
                    "response_text": "",
                },
            }
        },
        "model-info": {
            "grpc": {
                "endpoint": "fmaas.GenerationService/ModelInfo",
                "header": "",
                "body": '{"model_id": "$model_name"}',
                "args": "proto=text-generation-inference/proto/generation.proto",
                "response_fields_map": {"response": ""},
            }
        },
    },
    "caikit-standalone-runtime": {"containers": ["kserve-container"]},
    "caikit-standalone-runtime-grpc": {"containers": ["kserve-container", "queue-proxy", "istio-proxy"]},
    "vllm-runtime": {
        "chat-completions": {
            "http": {
                "endpoint": "v1/chat/completions",
                "header": "Content-Type:application/json",
                "body": '{"model": "$model_name","messages": [$query_text]}',
                "response_fields_map": {
                    "response": "choices",
                    "completion_tokens": "completion_tokens",
                    "response_text": "content",
                },
            }
        },
        "completions": {
            "http": {
                "endpoint": "v1/completions",
                "header": "Content-Type:application/json",
                "body": '{"model": "$model_name","prompt": "$query_text"}',
                "response_fields_map": {
                    "response": "choices",
                    "completion_tokens": "completion_tokens",
                    "response_text": "text",
                },
            }
        },
        "embeddings": {
            "http": {
                "endpoint": "v1/embeddings",
                "header": "Content-Type:application/json",
                "body": '{"encoding_format": "float", "model": "$model_name","input": "${query_text}"}',
                "response_fields_map": {
                    "response": "data",
                    "completion_tokens": "completion_tokens",
                    "response_text": "embedding",
                },
            }
        },
    },
}
