"""Worker request IDs stay unique across turns, conversations, and warmup."""

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.datasets.base import Turn
from evalscope.perf.utils.benchmark_util import BenchmarkData


def _run_requests(extra_args: dict[str, Any] | None) -> list[dict[str, Any]]:
    args = Arguments(
        model='test-model', api='openai', number=6, parallel=3,
        warmup_num=3, multi_turn=True, rate=-1, extra_args=extra_args,
    )
    args.number, args.parallel, args.rate = 6, 3, -1
    requests = []
    conversations = [
        [Turn(messages=[{'role': 'user', 'content': f'conv-{index}-turn-{turn}'}]) for turn in range(2)]
        for index in range(args.total_count)
    ]

    async def post(request: dict[str, Any]) -> BenchmarkData:
        requests.append(request)
        await asyncio.sleep(0)
        return BenchmarkData(success=True, prompt_tokens=8, completion_tokens=2, generated_text='ok')

    async def run() -> None:
        strategy = MultiTurnStrategy(
            args, OpenaiPlugin(args), SimpleNamespace(post=post), asyncio.Queue(), conversations,
        )
        await strategy.run()

    asyncio.run(run())
    return requests


def test_worker_ids_span_turns_conversations_and_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALSCOPE_MULTITURN_WORKER_RID_PREFIX', 'test-run-cc3')
    requests = _run_requests(None)
    assert len(requests) == 18
    assert len({request['rid'] for request in requests}) == len(requests)

    counters = defaultdict(list)
    conversation_workers = defaultdict(set)
    for request in requests:
        worker, counter = request['rid'].rsplit('_t', 1)
        counters[worker].append(int(counter))
        conversation_workers[request['messages'][0]['content']].add(worker)

    assert set(counters) == {f'test-run-cc3-slot{worker}' for worker in range(3)}
    for values in counters.values():
        assert values == list(range(1, len(values) + 1))
        assert len(values) > 2
    assert len(conversation_workers) == 9
    assert all(len(workers) == 1 for workers in conversation_workers.values())


def test_worker_ids_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('EVALSCOPE_MULTITURN_WORKER_RID_PREFIX', raising=False)
    assert all('rid' not in request for request in _run_requests(None))


def test_worker_ids_preserve_explicit_request_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALSCOPE_MULTITURN_WORKER_RID_PREFIX', 'test-run-cc3')
    requests = _run_requests({'rid': 'caller-request-id'})
    assert all(request['rid'] == 'caller-request-id' for request in requests)
