# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for warmup -> benchmark phase barrier in perf strategies.

These tests exercise ``ClosedLoopStrategy``, ``OpenLoopStrategy`` and
``MultiTurnStrategy`` directly with a fake HTTP client and fake API plugin,
asserting that every warmup request completes and lands in the queue
**before** any benchmark request is ever dispatched.  No real network
traffic is involved.
"""
import asyncio
import unittest
from typing import List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies import ClosedLoopStrategy, MultiTurnStrategy, OpenLoopStrategy
from evalscope.perf.core.strategies.base import BenchmarkStrategy
from evalscope.perf.utils.benchmark_util import BenchmarkData

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeApiPlugin:
    """Minimal ApiPluginBase stand-in for strategies that only call build_request."""

    def build_request(self, messages):
        # Identity-ish: return a dict that echoes the context length.
        return {'messages': list(messages)}

    def parse_responses(self, response_messages, request=None):  # pragma: no cover
        return (0, 0)


class _FakeClient:
    """Fake AioHttpClient whose ``post`` just sleeps a bit and returns a tagged BenchmarkData.

    The ``FakeClient`` assigns a monotonically increasing ``dispatch_order`` to
    each request the strategy fires, and a ``completion_order`` when the
    response resolves.  Tests inspect these counters to prove the phase
    barrier holds.
    """

    def __init__(self, *, warmup_delay: float = 0.02, benchmark_delay: float = 0.002):
        self._warmup_delay = warmup_delay
        self._benchmark_delay = benchmark_delay
        self._dispatch_counter = 0
        self._completion_counter = 0
        self.records: List[dict] = []
        # Per-conversation turn counter so multi-turn fake can produce
        # deterministic token counts.
        self._turn_no = 0

    # The strategies use ``async with client``; mimic AioHttpClient API.
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, body) -> BenchmarkData:
        self._dispatch_counter += 1
        my_dispatch = self._dispatch_counter

        # Default to benchmark delay; strategies may tag warmup via field later.
        # We discover warmup-ness via a convention: request bodies injected by
        # our test helpers contain '_is_warmup' in the messages list.
        is_warmup = bool(body.get('_is_warmup', False))
        await asyncio.sleep(self._warmup_delay if is_warmup else self._benchmark_delay)

        self._completion_counter += 1
        my_completion = self._completion_counter

        data = BenchmarkData(success=True)
        data.prompt_tokens = 1
        data.completion_tokens = 1
        data.query_latency = 0.0
        data.first_chunk_latency = 0.0
        self.records.append({
            'dispatch_order': my_dispatch,
            'completion_order': my_completion,
            'is_warmup_hint': is_warmup,
        })
        # Stash on the data object for tests to inspect.
        data._test_dispatch_order = my_dispatch
        data._test_completion_order = my_completion
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_queue(queue: asyncio.Queue) -> List[BenchmarkData]:
    """Drain a queue immediately and return all items in arrival order."""
    items: List[BenchmarkData] = []
    while not queue.empty():
        items.append(queue.get_nowait())
        queue.task_done()
    return items


async def _single_turn_generator(
    warmup_count: int, benchmark_count: int
):
    """Yield (request_dict, is_warmup) tuples emulating ``get_requests``."""
    for i in range(warmup_count):
        yield {'_is_warmup': True, 'idx': i}, True
    for i in range(benchmark_count):
        yield {'_is_warmup': False, 'idx': warmup_count + i}, False


def _mk_args(*, parallel=4, number=8, warmup_num=4, rate=-1, open_loop=False):
    args = Arguments(
        model='test-model',
        api='openai',
        url='http://127.0.0.1:9/v1/chat/completions',
        number=number,
        parallel=parallel,
        warmup_num=warmup_num,
        rate=rate,
        open_loop=open_loop,
    )
    # Mirror the list -> scalar coercion that ``run_perf_benchmark`` performs
    # before handing ``Arguments`` to the strategy layer.
    if isinstance(args.parallel, list):
        args.parallel = args.parallel[0]
    if isinstance(args.number, list):
        args.number = args.number[0]
    if isinstance(args.rate, list):
        args.rate = args.rate[0]
    return args


# ---------------------------------------------------------------------------
# Tests for the phase barrier at the strategy level
# ---------------------------------------------------------------------------


def _assert_phase_barrier(testcase: unittest.TestCase, items: List[BenchmarkData]) -> None:
    """All warmup items must have smaller dispatch/completion order than any benchmark item."""
    warmup_disp = [d._test_dispatch_order for d in items if d.is_warmup]
    benchmark_disp = [d._test_dispatch_order for d in items if not d.is_warmup]
    warmup_comp = [d._test_completion_order for d in items if d.is_warmup]
    benchmark_comp = [d._test_completion_order for d in items if not d.is_warmup]

    testcase.assertTrue(warmup_disp, 'expected at least one warmup item')
    testcase.assertTrue(benchmark_disp, 'expected at least one benchmark item')

    # Every warmup request was DISPATCHED before any benchmark request.
    testcase.assertLess(
        max(warmup_disp), min(benchmark_disp),
        'benchmark request was dispatched before warmup finished'
    )
    # Every warmup request COMPLETED before any benchmark request completed.
    testcase.assertLess(
        max(warmup_comp), min(benchmark_comp),
        'benchmark request completed before warmup finished'
    )


class TestClosedLoopPhaseBarrier(unittest.TestCase):

    def test_warmup_strictly_precedes_benchmark(self):
        async def _go():
            args = _mk_args(parallel=4, number=6, warmup_num=4)
            queue: asyncio.Queue = asyncio.Queue()
            client = _FakeClient(warmup_delay=0.05, benchmark_delay=0.001)
            gen = _single_turn_generator(warmup_count=args.warmup_count, benchmark_count=args.number)
            strategy = ClosedLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)
            await strategy.run()
            return await _collect_queue(queue)

        items = asyncio.run(_go())
        self.assertEqual(len([d for d in items if d.is_warmup]), 4)
        self.assertEqual(len([d for d in items if not d.is_warmup]), 6)
        _assert_phase_barrier(self, items)

    def test_no_warmup_runs_benchmark_only(self):
        async def _go():
            args = _mk_args(parallel=2, number=5, warmup_num=0)
            queue: asyncio.Queue = asyncio.Queue()
            client = _FakeClient()
            gen = _single_turn_generator(warmup_count=0, benchmark_count=args.number)
            strategy = ClosedLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)
            await strategy.run()
            return await _collect_queue(queue)

        items = asyncio.run(_go())
        self.assertEqual(len(items), 5)
        self.assertTrue(all(not d.is_warmup for d in items))


class TestOpenLoopPhaseBarrier(unittest.TestCase):

    def test_warmup_strictly_precedes_benchmark(self):
        async def _go():
            # rate must be > 0 in open-loop; use a modest rate so the test
            # is fast but Poisson sleeps are exercised.
            args = _mk_args(parallel=-1, number=6, warmup_num=3, rate=50.0, open_loop=True)
            queue: asyncio.Queue = asyncio.Queue()
            client = _FakeClient(warmup_delay=0.05, benchmark_delay=0.001)
            gen = _single_turn_generator(warmup_count=args.warmup_count, benchmark_count=args.number)
            strategy = OpenLoopStrategy(args, _FakeApiPlugin(), client, queue, gen)
            await strategy.run()
            return await _collect_queue(queue)

        items = asyncio.run(_go())
        self.assertEqual(len([d for d in items if d.is_warmup]), 3)
        self.assertEqual(len([d for d in items if not d.is_warmup]), 6)
        _assert_phase_barrier(self, items)


# ---------------------------------------------------------------------------
# Multi-turn tests
# ---------------------------------------------------------------------------


class _MultiTurnFakeClient(_FakeClient):
    """Multi-turn aware fake: records the context length passed to each post.

    We piggy-back on ``body['messages']`` (populated by
    ``_FakeApiPlugin.build_request``) to figure out which conversation index
    this turn belongs to.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.posted_conv_indices: List[int] = []

    async def post(self, body) -> BenchmarkData:
        # Conversation index is carried in the first message payload.
        msgs = body['messages']
        # Search backwards for the deepest 'conv_idx' marker.
        conv_idx = None
        for m in msgs:
            if isinstance(m, dict) and 'conv_idx' in m:
                conv_idx = m['conv_idx']
        self.posted_conv_indices.append(conv_idx if conv_idx is not None else -1)
        return await super().post(body)


def _mk_conversations(total: int, turns_per_conv: int = 2):
    """Build ``total`` synthetic conversations, each with ``turns_per_conv`` turns.

    Every user message carries its ``conv_idx`` so the fake client can attribute
    each turn back to its originating conversation.
    """
    convs = []
    for ci in range(total):
        convs.append([
            [{'role': 'user', 'content': f'conv{ci}-t{ti}', 'conv_idx': ci}]
            for ti in range(turns_per_conv)
        ])
    return convs


class TestMultiTurnPhaseBarrier(unittest.TestCase):

    def test_warmup_conversations_complete_before_benchmark(self):
        async def _go():
            args = _mk_args(parallel=2, number=4, warmup_num=2)
            queue: asyncio.Queue = asyncio.Queue()
            client = _MultiTurnFakeClient(warmup_delay=0.03, benchmark_delay=0.001)
            conversations = _mk_conversations(total=8, turns_per_conv=2)
            strategy = MultiTurnStrategy(args, _FakeApiPlugin(), client, queue, conversations)
            await strategy.run()
            return await _collect_queue(queue), client

        items, client = asyncio.run(_go())

        warmup_items = [d for d in items if d.is_warmup]
        benchmark_items = [d for d in items if not d.is_warmup]

        # 2 warmup convs * 2 turns and 4 benchmark convs * 2 turns.
        self.assertEqual(len(warmup_items), 4)
        self.assertEqual(len(benchmark_items), 8)

        # Phase barrier: every warmup turn was dispatched and completed
        # before any benchmark turn.
        _assert_phase_barrier(self, items)

        # Conversation index disjointness: warmup consumed indices [0, 1],
        # benchmark consumed [2, 3, 4, 5].
        warmup_conv_indices = set()
        benchmark_conv_indices = set()
        # Re-derive from client records aligned with queue items.  Since items
        # pop in queue order (warmup first, then benchmark) and client
        # posted_conv_indices is also in completion order, we align by
        # is_warmup flag.
        warmup_count_seen = 0
        for idx, d in enumerate(items):
            if d.is_warmup:
                warmup_conv_indices.add(client.posted_conv_indices[idx])
                warmup_count_seen += 1
            else:
                benchmark_conv_indices.add(client.posted_conv_indices[idx])
        self.assertEqual(warmup_conv_indices, {0, 1})
        self.assertEqual(benchmark_conv_indices, {2, 3, 4, 5})

    def test_no_warmup_runs_benchmark_only(self):
        async def _go():
            args = _mk_args(parallel=2, number=3, warmup_num=0)
            queue: asyncio.Queue = asyncio.Queue()
            client = _MultiTurnFakeClient()
            conversations = _mk_conversations(total=5, turns_per_conv=2)
            strategy = MultiTurnStrategy(args, _FakeApiPlugin(), client, queue, conversations)
            await strategy.run()
            return await _collect_queue(queue)

        items = asyncio.run(_go())
        self.assertTrue(all(not d.is_warmup for d in items))
        self.assertEqual(len(items), 6)  # 3 convs * 2 turns


if __name__ == '__main__':
    unittest.main(buffer=False)
