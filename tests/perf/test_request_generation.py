import asyncio
import numpy as np
from pytest import MonkeyPatch
from typing import Dict, Iterator, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.benchmark import _resolve_request_generation_workers, get_requests
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.random_dataset import RandomDatasetPlugin
from evalscope.perf.plugin.registry import DatasetRegistry


class _ParallelDatasetPlugin(DatasetPluginBase):
    last_workers: int = 0

    def build_messages(self) -> Iterator[str]:
        raise AssertionError('serial generation should not be used')

    def supports_parallel_message_generation(self) -> bool:
        return True

    def build_messages_parallel(self, total_count: int, workers: int) -> List[str]:
        type(self).last_workers = workers
        return [f'message-{index}' for index in range(total_count)]


class _FakeApiPlugin:

    def build_request(self, messages: str) -> Dict[str, str]:
        return {'payload': messages}


class _ForkContext:

    def get_start_method(self) -> str:
        return 'fork'


async def _collect_requests(args: Arguments) -> List:
    return [item async for item in get_requests(args, _FakeApiPlugin())]


def _make_args(**kwargs) -> Arguments:
    params = {
        'model': 'test-model',
        'url': 'http://127.0.0.1:8000/v1/chat/completions',
        'dataset': 'unit_parallel_dataset',
        'number': 3,
        'parallel': 1,
        'request_generation_workers': 2,
    }
    params.update(kwargs)
    return Arguments(**params)


def test_parallel_dataset_generation_hook_preserves_order_and_warmup() -> None:
    DatasetRegistry.register('unit_parallel_dataset', _ParallelDatasetPlugin)
    args = _make_args(warmup_num=1)

    requests = asyncio.run(_collect_requests(args))

    assert _ParallelDatasetPlugin.last_workers == 2
    assert requests == [
        ({'payload': 'message-0'}, True),
        ({'payload': 'message-1'}, False),
        ({'payload': 'message-2'}, False),
        ({'payload': 'message-3'}, False),
    ]


def test_request_generation_worker_auto_respects_cpu_affinity(monkeypatch: MonkeyPatch) -> None:
    args = _make_args(number=256, request_generation_workers=0)
    monkeypatch.setattr('evalscope.perf.benchmark.os.sched_getaffinity', lambda _: {0, 1, 2, 3}, raising=False)

    workers = _resolve_request_generation_workers(args, total_count=256, supports_parallel_generation=True)

    assert workers == 4


def test_request_generation_worker_auto_amortizes_small_runs(monkeypatch: MonkeyPatch) -> None:
    args = _make_args(number=24, request_generation_workers=0)
    monkeypatch.setattr('evalscope.perf.benchmark.os.sched_getaffinity', lambda _: set(range(128)), raising=False)

    workers = _resolve_request_generation_workers(args, total_count=24, supports_parallel_generation=True)

    assert workers == 1


def test_request_generation_worker_auto_is_capped(monkeypatch: MonkeyPatch) -> None:
    args = _make_args(number=4096, request_generation_workers=0)
    monkeypatch.setattr('evalscope.perf.benchmark.os.sched_getaffinity', lambda _: set(range(128)), raising=False)

    workers = _resolve_request_generation_workers(args, total_count=4096, supports_parallel_generation=True)

    assert workers == 32


def test_request_generation_workers_can_disable_parallel_path() -> None:
    args = _make_args(number=10, request_generation_workers=1)

    workers = _resolve_request_generation_workers(args, total_count=10, supports_parallel_generation=True)

    assert workers == 1


def test_random_dataset_auto_parallel_requires_long_prompts(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr('evalscope.perf.plugin.datasets.random_dataset._get_random_generation_context',
                        lambda: _ForkContext())

    short_args = _make_args(
        dataset='random',
        number=512,
        request_generation_workers=0,
        min_prompt_length=64,
        max_prompt_length=64,
        tokenize_prompt=False,
    )
    short_plugin = object.__new__(RandomDatasetPlugin)
    short_plugin.query_parameters = short_args

    mid_args = _make_args(
        dataset='random',
        number=512,
        request_generation_workers=0,
        min_prompt_length=1024,
        max_prompt_length=1024,
        tokenize_prompt=False,
    )
    mid_plugin = object.__new__(RandomDatasetPlugin)
    mid_plugin.query_parameters = mid_args

    long_args = _make_args(
        dataset='random',
        number=512,
        request_generation_workers=0,
        min_prompt_length=2048,
        max_prompt_length=2048,
        tokenize_prompt=False,
    )
    long_plugin = object.__new__(RandomDatasetPlugin)
    long_plugin.query_parameters = long_args

    explicit_args = _make_args(
        dataset='random',
        number=512,
        request_generation_workers=2,
        min_prompt_length=64,
        max_prompt_length=64,
        tokenize_prompt=False,
    )
    explicit_plugin = object.__new__(RandomDatasetPlugin)
    explicit_plugin.query_parameters = explicit_args

    assert not short_plugin.supports_parallel_message_generation()
    assert not mid_plugin.supports_parallel_message_generation()
    assert long_plugin.supports_parallel_message_generation()
    assert explicit_plugin.supports_parallel_message_generation()


def test_random_dataset_parallel_requires_fork_safe_context(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr('evalscope.perf.plugin.datasets.random_dataset._get_random_generation_context', lambda: None)

    args = _make_args(
        dataset='random',
        number=512,
        request_generation_workers=2,
        min_prompt_length=2048,
        max_prompt_length=2048,
        tokenize_prompt=False,
    )
    plugin = object.__new__(RandomDatasetPlugin)
    plugin.query_parameters = args
    plugin.number = args.total_count
    plugin.build_messages = lambda: iter(['serial-0', 'serial-1'])

    assert not plugin.supports_parallel_message_generation()
    assert plugin.build_messages_parallel(total_count=2, workers=2) == ['serial-0', 'serial-1']


def test_random_dataset_serial_generation_uses_item_local_seeds(monkeypatch: MonkeyPatch) -> None:
    def fake_generate_token_sequence(
        self: RandomDatasetPlugin,
        input_len: int,
        offset: int,
        index: int,
    ) -> Tuple[str, int, int]:
        return f'{input_len}-{offset}-{index}-{np.random.randint(0, 100000)}', input_len, 0

    monkeypatch.setattr(RandomDatasetPlugin, 'generate_token_sequence', fake_generate_token_sequence)

    args = _make_args(
        dataset='random',
        number=8,
        request_generation_workers=1,
        min_prompt_length=4,
        max_prompt_length=4,
        apply_chat_template=False,
        tokenize_prompt=False,
    )

    np.random.seed(123)
    serial_plugin = object.__new__(RandomDatasetPlugin)
    serial_plugin.query_parameters = args
    serial_plugin.number = args.total_count
    serial_plugin.allowed_tokens = np.arange(100)
    serial_plugin.prefix_ids = []
    serial_plugin.prefix_length = 0
    serial = list(serial_plugin.build_messages())

    np.random.seed(123)
    expected_plugin = object.__new__(RandomDatasetPlugin)
    expected_plugin.query_parameters = args
    expected_plugin.number = args.total_count
    expected_plugin.allowed_tokens = np.arange(100)
    expected_plugin.prefix_ids = []
    expected_plugin.prefix_length = 0
    plan = expected_plugin._create_generation_plan(args.total_count, include_seeds=True)
    expected = [
        expected_plugin._build_random_message(input_len, offset, index, seed)[0]
        for index, input_len, offset, seed in expected_plugin._iter_generation_plan(plan)
    ]

    assert serial == expected
