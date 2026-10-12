"""Dataset conversation IDs propagate to requests across workers and warmup."""

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.datasets.swe_smith import SweSmithDatasetPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    conversations = []
    for index in range(9):
        conversations.append([
            {
                'rid': f'conversation-{index}',
                'messages': [{'role': 'user', 'content': f'[rid:cache-{index}] first turn'}],
            },
            {'messages': [{'role': 'user', 'content': 'second turn'}]},
        ])
    path = tmp_path / 'conversations.json'
    path.write_text(json.dumps({'conversations': conversations}))
    return path


def _run_requests(dataset_path: Path, extra_args: dict[str, Any] | None) -> list[dict[str, Any]]:
    args = Arguments(
        model='test-model', api='openai', dataset='swe_smith', dataset_path=str(dataset_path),
        number=6, parallel=3, warmup_num=3, multi_turn=True, rate=-1, extra_args=extra_args,
    )
    args.number, args.parallel, args.rate = 6, 3, -1
    requests = []
    conversations = list(SweSmithDatasetPlugin(args).build_messages())

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


def test_conversation_ids_span_turns_and_remain_distinct(dataset_path: Path) -> None:
    requests = _run_requests(dataset_path, None)
    assert len(requests) == 18
    assert len({request['rid'] for request in requests}) == len(requests)
    by_conversation = defaultdict(list)
    for request in requests:
        by_conversation[request['messages'][0]['content']].append(request['rid'])
    assert by_conversation == {
        f'[rid:cache-{index}] first turn': [f'conversation-{index}_t1', f'conversation-{index}_t2']
        for index in range(9)
    }


def test_prompt_markers_do_not_supply_request_ids(dataset_path: Path) -> None:
    data = json.loads(dataset_path.read_text())
    for conversation in data['conversations']:
        conversation[0].pop('rid')
    dataset_path.write_text(json.dumps(data))
    assert all('rid' not in request for request in _run_requests(dataset_path, None))


def test_explicit_request_ids_are_preserved(dataset_path: Path) -> None:
    requests = _run_requests(dataset_path, {'rid': 'caller-request-id'})
    assert all(request['rid'] == 'caller-request-id' for request in requests)
