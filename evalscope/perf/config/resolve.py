from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict
from typing import List

from evalscope.perf.config.models import ConversationLoad, LoadSpec, PerfConfig
from evalscope.perf.domain.errors import PerfConfigError


class ResolvedRunSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    load_id: str
    load: LoadSpec
    seed: int
    warmup_count: int
    item_limit: int | None


class ResolvedSuite(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: PerfConfig
    runs: List[ResolvedRunSpec]


def resolve_suite(config: PerfConfig) -> ResolvedSuite:
    """Validate cross-component capabilities and derive immutable run specs."""
    try:
        from evalscope.perf.protocols import protocol_registry
        from evalscope.perf.workloads.registry import workload_registry

        protocol_registry.get(config.target.protocol)
        workload_cls = workload_registry.get(config.workload.name)
        meta = workload_cls.meta
    except Exception as e:
        raise PerfConfigError(str(e)) from e

    if config.target.protocol not in meta.protocols:
        raise PerfConfigError(
            f'Workload {meta.name!r} does not support protocol {config.target.protocol!r}; '
            f'supported protocols: {sorted(meta.protocols)}'
        )
    if config.workload.path and not meta.supports_local_path:
        raise PerfConfigError(f'Workload {meta.name!r} does not support a local path')
    if config.workload.path and not os.path.exists(config.workload.path):
        raise PerfConfigError(f'Workload path {config.workload.path!r} does not exist')
    if not config.workload.path:
        if config.workload.data_source == 'local' and meta.requires_dataset:
            raise PerfConfigError('A workload path is required when data_source is local')
        if config.workload.data_source == 'modelscope' and meta.requires_dataset and not meta.supports_modelscope:
            raise PerfConfigError(f'Workload {meta.name!r} does not support ModelScope data')
        if config.workload.data_source == 'huggingface' and meta.requires_dataset and not meta.supports_huggingface:
            raise PerfConfigError(f'Workload {meta.name!r} does not support Hugging Face data')
    if meta.requires_tokenizer and not config.target.tokenizer:
        raise PerfConfigError(f'Workload {meta.name!r} requires target.tokenizer')

    expected_endpoint = {
        'openai_chat': 'chat/completions',
        'openai_completions': 'completions',
        'openai_responses': 'responses',
        'openai_embedding': 'embeddings',
        'openai_rerank': 'reranks',
    }.get(config.target.protocol)
    known_endpoints = ('chat/completions', 'completions', 'responses', 'embeddings', 'reranks')
    matched = next((item for item in known_endpoints if config.target.base_url.rstrip('/').endswith(item)), None)
    if expected_endpoint and matched and matched != expected_endpoint:
        raise PerfConfigError(
            f'Target URL ends with {matched!r}, but protocol {config.target.protocol!r} requires {expected_endpoint!r}'
        )

    resolved = []
    base_seed = config.runtime.seed or 0
    for index, load in enumerate(config.suite.loads):
        is_conversation = isinstance(load, ConversationLoad)
        expected_mode = 'conversation' if is_conversation else 'single_turn'
        if meta.mode != expected_mode:
            raise PerfConfigError(
                f'Load mode {load.mode!r} requires a {expected_mode} workload, but {meta.name!r} is {meta.mode}'
            )
        count = load.conversation_count if is_conversation else load.request_count
        warmup_base = count or 1
        warmup_count = load.warmup.resolve(warmup_base)
        resolved.append(
            ResolvedRunSpec(
                load_id=f'{index:03d}-{load.mode}',
                load=load,
                seed=base_seed + index,
                warmup_count=warmup_count,
                item_limit=count,
            )
        )
    return ResolvedSuite(config=config, runs=resolved)
