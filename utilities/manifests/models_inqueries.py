from typing import Any, Dict

INQUIRIES: Dict[str, Any] = {
    "water_boil": {
        "query_text": "At what temperature does water boil?",
        "models": {
            "flan-t5-small-caikit": {
                "response_tokens": 5,
                "response_text": "74 degrees F",
                "streamed_response_text": [
                    {"details": {"input_token_count": "8"}},
                    {
                        "tokens": [{"text": "▁", "logprob": -1.6961838006973267}],
                        "details": {"generated_tokens": 1},
                    },
                    {
                        "generated_text": "74",
                        "tokens": [{"text": "74", "logprob": -3.250730037689209}],
                        "details": {"generated_tokens": 2},
                    },
                    {
                        "generated_text": "degrees",
                        "tokens": [{"text": "▁degrees", "logprob": -0.4324559271335602}],
                        "details": {"generated_tokens": 3},
                    },
                    {
                        "generated_text": "F",
                        "tokens": [{"text": "▁F", "logprob": -1.361091136932373}],
                        "details": {"generated_tokens": 4},
                    },
                    {
                        "tokens": [{"text": "\u003c/s\u003e", "logprob": -0.010431881994009018}],
                        "details": {
                            "finish_reason": "EOS_TOKEN",
                            "generated_tokens": 5,
                        },
                    },
                ],
            },
            "tgis-runtime": {
                "tokenize_response_text": {
                    "responses": [
                        {
                            "tokenCount": 8,
                            "tokens": [
                                "▁At",
                                "▁what",
                                "▁temperature",
                                "▁does",
                                "▁water",
                                "▁boil",
                                "?",
                                "\u003c/s\u003e",
                            ],
                        }
                    ]
                }
            },
        },
    },
}
