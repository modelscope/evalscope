import asyncio
import pytest
from pydantic import ValidationError

from evalscope.perf import (
    BenchmarkSuite,
    ClosedLoopLoad,
    ConversationLoad,
    OpenLoopLoad,
    PerfConfig,
    TargetConfig,
    WarmupConfig,
    WorkloadConfig,
    async_run_perf,
)
from evalscope.perf.config import resolve_suite
from evalscope.perf.domain.errors import PerfConfigError
from evalscope.perf.domain.workload import WorkloadMeta
from evalscope.perf.workloads.base import WorkloadSource
from evalscope.perf.workloads.registry import WorkloadRegistry


def test_load_limits_are_required() -> None:
    with pytest.raises(ValidationError):
        ClosedLoopLoad()
    with pytest.raises(ValidationError):
        ConversationLoad()
    with pytest.raises(ValidationError):
        OpenLoopLoad(request_rate=1)


def test_warmup_count_and_ratio_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError):
        WarmupConfig(count=1, ratio=0.1)


def test_conversation_load_rejects_single_turn_workload() -> None:
    config = PerfConfig(
        target=TargetConfig(model='model'),
        workload=WorkloadConfig(name='openqa'),
        suite=BenchmarkSuite(loads=[ConversationLoad(conversation_count=1)]),
    )
    with pytest.raises(PerfConfigError, match='requires a conversation workload'):
        resolve_suite(config)


def test_conversation_rejects_tokenized_completion_protocol() -> None:
    config = PerfConfig(
        target=TargetConfig(model='model', protocol='openai_completions'),
        workload=WorkloadConfig(name='custom_multi_turn', path=__file__, data_source='local'),
        suite=BenchmarkSuite(loads=[ConversationLoad(conversation_count=1)]),
    )
    with pytest.raises(PerfConfigError, match='does not support protocol'):
        resolve_suite(config)


def test_open_loop_has_explicit_bounded_outstanding() -> None:
    load = OpenLoopLoad(request_rate=10, request_count=100, max_outstanding=7)
    assert load.max_outstanding == 7
    assert not hasattr(load, 'concurrency')


def test_registry_rejects_duplicate_workloads() -> None:

    class Source(WorkloadSource):
        meta = WorkloadMeta(name='duplicate', mode='single_turn')

        async def iter_items(self, run):
            if False:
                yield

    registry = WorkloadRegistry()
    registry.register(Source)
    with pytest.raises(PerfConfigError, match='already registered'):
        registry.register(Source)


def test_mapping_validation_uses_perf_error() -> None:

    async def invalid():
        await async_run_perf({'target': {'model': 'fake'}, 'suite': {'loads': []}})

    with pytest.raises(PerfConfigError):
        asyncio.run(invalid())


def test_target_secrets_are_not_serialized() -> None:
    target = TargetConfig(
        model='fake',
        api_key='secret',
        headers={
            'Authorization': 'Bearer secret',
            'X-Trace': 'visible'
        },
    )
    payload = target.model_dump(mode='json')
    assert 'api_key' not in payload
    assert payload['headers'] == {'Authorization': '***', 'X-Trace': 'visible'}
