# Copyright (c) Alibaba, Inc. and its affiliates.
"""K2-spec parameter contract for vendor compliance probing.

A vendor that hosts Kimi K2 / K2-Thinking MUST hard-fix these decoding
parameters to the model's recommended defaults; any other value must be
rejected with HTTP 400. This file is a faithful port of
``IMMUTABLE_PARAMS`` from Kimi-Vendor-Verifier's ``verify_params.py``.
"""
from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class ParamSpec:
    name: str
    think_default: Any
    non_think_default: Any
    wrong_value: Any


IMMUTABLE_PARAMS: List[ParamSpec] = [
    ParamSpec('temperature', 1.0, 0.6, 0.5),
    ParamSpec('top_p', 0.95, 0.95, 0.8),
    ParamSpec('presence_penalty', 0, 0, 0.5),
    ParamSpec('frequency_penalty', 0, 0, 0.5),
    ParamSpec('n', 1, 1, 2),
]


def thinking_extra_body(thinking: bool, think_mode: str) -> dict:
    """Build the ``extra_body`` payload for the given thinking mode.

    - ``kimi``: official Moonshot SaaS API contract.
    - ``opensource``: vLLM/SGLang/KTransformers HuggingFace chat template hook.
    - ``none``: non-hybrid model; no thinking parameter.
    """
    if think_mode == 'none':
        return {}
    if think_mode == 'opensource':
        return {'chat_template_kwargs': {'thinking': thinking}}
    return {'thinking': {'type': 'enabled' if thinking else 'disabled'}}
