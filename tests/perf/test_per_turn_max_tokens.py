"""Per-turn output limits and custom multi-turn system prompt coverage."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from evalscope.perf.arguments import Arguments, add_argument
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy
from evalscope.perf.plugin.api.dashscope_api import DashScopeApiPlugin
from evalscope.perf.plugin.api.openai_responses_api import OpenAIResponsesPlugin
from evalscope.perf.plugin.datasets.base import Turn
from evalscope.perf.plugin.datasets.custom import CustomMultiTurnDatasetPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData


def _make_args(**kwargs: Any) -> Arguments:
    number = kwargs.pop('number', 1)
    parallel = kwargs.pop('parallel', 1)
    args = Arguments(
        model='test-model',
        number=number,
        parallel=parallel,
        rate=-1,
        multi_turn=True,
        **kwargs,
    )
    args.number = number
    args.parallel = parallel
    args.rate = -1
    return args


class _ApiPlugin:

    def build_request(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {'messages': messages, 'max_tokens': 2048}


class _RecordingClient:

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []

    async def post(self, request: Dict[str, Any]) -> BenchmarkData:
        self.requests.append(request)
        return BenchmarkData(
            request=json.dumps(request),
            success=True,
            generated_text='ok',
            prompt_tokens=1,
            completion_tokens=1,
        )


def _run_conversation(args: Arguments, turns: List[Turn]) -> List[Dict[str, Any]]:
    client = _RecordingClient()

    async def run() -> None:
        strategy = MultiTurnStrategy(args, _ApiPlugin(), client, asyncio.Queue(), [turns])
        await strategy.run()

    asyncio.run(run())
    return client.requests


def test_max_turn_tokens_reuses_final_value_and_overrides_trace_values() -> None:
    args = _make_args(max_turn_tokens=[150, 1000])
    turns = [Turn(messages=[{'role': 'user', 'content': str(i)}], max_tokens=10 + i) for i in range(3)]

    requests = _run_conversation(args, turns)

    assert [request['max_tokens'] for request in requests] == [150, 1000, 1000]


def test_trace_max_tokens_remain_effective_without_cli_override() -> None:
    args = _make_args()
    turns = [Turn(messages=[{'role': 'user', 'content': str(i)}], max_tokens=value) for i, value in enumerate([10, 20])]

    requests = _run_conversation(args, turns)

    assert [request['max_tokens'] for request in requests] == [10, 20]


@pytest.mark.parametrize('value, expected', [(120, [120]), ([120, 240], [120, 240])])
def test_max_turn_tokens_validation(value: Any, expected: List[int]) -> None:
    assert _make_args(max_turn_tokens=value).max_turn_tokens == expected


def test_max_turn_tokens_cli_parsing() -> None:
    parser = argparse.ArgumentParser()
    add_argument(parser)

    parsed = parser.parse_args(['--model', 'test-model', '--multi-turn', '--max-turn-tokens', '150', '1000'])

    assert parsed.max_turn_tokens == [150, 1000]


@pytest.mark.parametrize('value', [[], [-1], [1.5], [True]])
def test_invalid_max_turn_tokens_rejected(value: Any) -> None:
    with pytest.raises(ValidationError, match='max-turn-tokens'):
        _make_args(max_turn_tokens=value)


def test_protocol_plugins_set_their_native_output_limit_fields() -> None:
    responses_request: Dict[str, Any] = {'max_output_tokens': 2048}
    OpenAIResponsesPlugin.set_request_max_tokens(None, responses_request, 150)
    assert responses_request == {'max_output_tokens': 150}

    dashscope_request: Dict[str, Any] = {'parameters': {'max_tokens': 2048}}
    DashScopeApiPlugin.set_request_max_tokens(None, dashscope_request, 150)
    assert dashscope_request == {'parameters': {'max_tokens': 150}}


def test_custom_multi_turn_preserves_system_prompt_in_first_turn(tmp_path: Path) -> None:
    dataset_path = tmp_path / 'conversations.jsonl'
    messages = [
        {'role': 'system', 'content': 'Use the configured agent tools.'},
        {'role': 'user', 'content': 'Start'},
        {'role': 'assistant', 'content': 'reference boundary'},
        {'role': 'user', 'content': 'Continue'},
    ]
    dataset_path.write_text(f'{json.dumps(messages)}\n', encoding='utf-8')
    args = _make_args(dataset='custom_multi_turn', dataset_path=str(dataset_path))

    conversation = next(iter(CustomMultiTurnDatasetPlugin(args)))

    assert conversation[0].messages == messages[:2]
    assert conversation[1].messages == [messages[3]]
