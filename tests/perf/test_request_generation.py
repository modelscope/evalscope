import argparse
import asyncio
import numpy as np
from pytest import MonkeyPatch
from typing import Dict, Iterator, List, Optional, Tuple

from evalscope.perf.arguments import Arguments, add_argument
from evalscope.perf.benchmark import get_requests
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.random_dataset import RandomDatasetPlugin, _get_random_generation_context
from evalscope.perf.plugin.registry import DatasetRegistry
from evalscope.perf.utils.worker_util import resolve_dataset_generation_workers

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_random_plugin(args: Arguments) -> RandomDatasetPlugin:
    """Create a lightweight RandomDatasetPlugin without running ``__init__``."""
    plugin = object.__new__(RandomDatasetPlugin)
    plugin.query_parameters = args
    plugin.number = args.total_count
    plugin.tokenizer = None
    plugin.allowed_tokens = np.arange(100)
    plugin.prefix_ids = []
    plugin.prefix_length = 0
    return plugin


class _ParallelDatasetPlugin(DatasetPluginBase):
    last_workers: int = 0

    def build_messages(self) -> Iterator[str]:
        raise AssertionError('serial generation should not be used')

    def supports_parallel_message_generation(self, total_count: Optional[int] = None) -> bool:
        return True

    def build_messages_parallel(self, total_count: int, workers: int) -> List[str]:
        type(self).last_workers = workers
        return [f'message-{index}' for index in range(total_count)]


class _FakeApiPlugin:

    def build_request(self, messages: str) -> Dict[str, str]:
        return {'payload': messages}


async def _collect_requests(args: Arguments) -> List:
    return [item async for item in get_requests(args, _FakeApiPlugin())]


def _make_args(**kwargs) -> Arguments:
    params = {
        'model': 'test-model',
        'url': 'http://127.0.0.1:8000/v1/chat/completions',
        'dataset': 'unit_parallel_dataset',
        'number': 3,
        'parallel': 1,
        'num_workers': 2,
    }
    params.update(kwargs)
    return Arguments(**params)


def _parse_perf_args(argv: List[str]) -> Arguments:
    parser = argparse.ArgumentParser()
    add_argument(parser)
    return Arguments.from_args(parser.parse_args(argv))


def test_sweep_scalar_strings_are_normalized_to_lists() -> None:
    args = _make_args(number='10', parallel='2')

    assert args.number == [10]
    assert args.parallel == [2]

    args = _make_args(number='10', rate='5.0', open_loop=True)

    assert args.number == [10]
    assert args.rate == [5.0]


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


def test_dataset_generation_worker_auto_respects_cpu_affinity(monkeypatch: MonkeyPatch) -> None:
    args = _make_args(number=512, num_workers=0)
    monkeypatch.setattr('evalscope.perf.utils.worker_util.os.sched_getaffinity', lambda _: {0, 1, 2, 3}, raising=False)

    workers = resolve_dataset_generation_workers(args, total_count=512, supports_parallel_generation=True)

    assert workers == 4


def test_dataset_generation_worker_auto_amortizes_small_runs(monkeypatch: MonkeyPatch) -> None:
    args = _make_args(number=127, num_workers=0)
    monkeypatch.setattr('evalscope.perf.utils.worker_util.os.sched_getaffinity', lambda _: set(range(128)), raising=False)

    workers = resolve_dataset_generation_workers(args, total_count=127, supports_parallel_generation=True)

    assert workers == 1


def test_dataset_generation_worker_auto_is_capped(monkeypatch: MonkeyPatch) -> None:
    args = _make_args(number=4096, num_workers=0)
    monkeypatch.setattr('evalscope.perf.utils.worker_util.os.sched_getaffinity', lambda _: set(range(128)), raising=False)

    workers = resolve_dataset_generation_workers(args, total_count=4096, supports_parallel_generation=True)

    assert workers == 32


def test_dataset_generation_workers_can_disable_parallel_path() -> None:
    args = _make_args(number=10, num_workers=1)

    workers = resolve_dataset_generation_workers(args, total_count=10, supports_parallel_generation=True)

    assert workers == 1


def test_multi_turn_num_workers_is_promoted_to_top_level() -> None:
    args = Arguments(
        model='test-model',
        url='http://127.0.0.1:8000/v1/chat/completions',
        dataset='unit_parallel_dataset',
        number=3,
        parallel=1,
        multi_turn_args={'num_workers': 3},
    )

    assert args.num_workers == 3


def test_top_level_num_workers_takes_precedence_over_multi_turn_value() -> None:
    args = _make_args(num_workers=0, multi_turn_args={'num_workers': 3})

    assert args.num_workers == 0


def test_apply_chat_template_boolean_cli_flags() -> None:
    default_args = _parse_perf_args(['--model', 'test-model', '--url', 'http://127.0.0.1:8000/v1/completions'])
    enabled_args = _parse_perf_args([
        '--model',
        'test-model',
        '--url',
        'http://127.0.0.1:8000/v1/completions',
        '--apply-chat-template',
    ])
    disabled_args = _parse_perf_args([
        '--model',
        'test-model',
        '--url',
        'http://127.0.0.1:8000/v1/chat/completions',
        '--no-apply-chat-template',
    ])

    assert default_args.apply_chat_template is False
    assert enabled_args.apply_chat_template is True
    assert disabled_args.apply_chat_template is False


def test_random_dataset_parallel_uses_spawn_context() -> None:
    assert _get_random_generation_context().get_start_method() == 'spawn'


def test_random_dataset_auto_parallel_requires_large_long_prompt_work() -> None:
    short_plugin = _make_random_plugin(_make_args(
        dataset='random', number=512, num_workers=0,
        min_prompt_length=64, max_prompt_length=64, tokenize_prompt=False,
    ))
    mid_plugin = _make_random_plugin(_make_args(
        dataset='random', number=512, num_workers=0,
        min_prompt_length=2048, max_prompt_length=2048, tokenize_prompt=False,
    ))
    small_long_plugin = _make_random_plugin(_make_args(
        dataset='random', number=128, num_workers=0,
        min_prompt_length=8192, max_prompt_length=8192, tokenize_prompt=False,
    ))
    large_long_plugin = _make_random_plugin(_make_args(
        dataset='random', number=512, num_workers=0,
        min_prompt_length=8192, max_prompt_length=8192, tokenize_prompt=False,
    ))
    explicit_plugin = _make_random_plugin(_make_args(
        dataset='random', number=512, num_workers=2,
        min_prompt_length=64, max_prompt_length=64, tokenize_prompt=False,
    ))

    assert not short_plugin.supports_parallel_message_generation()
    assert not mid_plugin.supports_parallel_message_generation()
    assert not small_long_plugin.supports_parallel_message_generation()
    assert large_long_plugin.supports_parallel_message_generation()
    assert explicit_plugin.supports_parallel_message_generation()


def test_random_dataset_serial_generation_uses_item_local_seeds(monkeypatch: MonkeyPatch) -> None:
    def fake_gen_prompt(tokenizer, token_sequence, target_token_len, add_special_tokens, allowed_tokens):
        """Return a prompt that embeds the current numpy random state so we can
        verify that seeds are applied before each item."""
        prompt = f'{target_token_len}-{np.random.randint(0, 100000)}'
        return prompt, token_sequence, 0

    monkeypatch.setattr(
        'evalscope.perf.plugin.datasets.random_dataset.gen_prompt_decode_to_target_len',
        fake_gen_prompt,
    )

    args = _make_args(
        dataset='random',
        number=8,
        num_workers=1,
        min_prompt_length=4,
        max_prompt_length=4,
        apply_chat_template=False,
        tokenize_prompt=False,
    )

    np.random.seed(123)
    serial_plugin = _make_random_plugin(args)
    serial = list(serial_plugin.build_messages())

    np.random.seed(123)
    expected_plugin = _make_random_plugin(args)
    plan = expected_plugin._create_generation_plan(args.total_count, include_seeds=True)
    expected = [
        expected_plugin._build_random_message(input_len, offset, index, seed)[0]
        for index, input_len, offset, seed in expected_plugin._iter_generation_plan(plan)
    ]

    assert serial == expected
