# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-turn max_tokens must land in each protocol's native field/location.

MultiTurnStrategy._worker routes the per-turn output cap through
ApiPluginBase.set_request_max_tokens; protocols that do not use a top-level
``max_tokens`` override it (DashScope nests it, OpenAI Responses renames it).
"""
import asyncio
import json
from typing import Any, Dict, List, Optional

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy
from evalscope.perf.plugin.api.dashscope_api import DashScopeApiPlugin
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.api.openai_responses_api import OpenAIResponsesPlugin
from evalscope.perf.plugin.datasets.base import Turn
from evalscope.perf.utils.benchmark_util import BenchmarkData


def test_set_request_max_tokens_uses_protocol_native_field() -> None:
    openai_req: Dict[str, Any] = {}
    OpenaiPlugin(Arguments(model='m', api='openai')).set_request_max_tokens(openai_req, 42)
    assert openai_req == {'max_tokens': 42}

    dashscope_req: Dict[str, Any] = {}
    DashScopeApiPlugin(Arguments(model='m', api='dashscope')).set_request_max_tokens(dashscope_req, 42)
    assert dashscope_req == {'parameters': {'max_tokens': 42}}

    responses_req: Dict[str, Any] = {}
    OpenAIResponsesPlugin(Arguments(model='m', api='openai_responses')).set_request_max_tokens(responses_req, 42)
    assert responses_req == {'max_output_tokens': 42}


class _CapPlugin:
    """Records the per-turn cap and nests it the way DashScope does."""

    def __init__(self) -> None:
        self.caps: List[int] = []

    def build_request(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {'messages': list(messages), 'stream': True}

    def parse_responses(self, response_messages: List[Any], request: Optional[str] = None) -> tuple[int, int]:
        return 1, 1

    def set_request_max_tokens(self, request: Dict[str, Any], max_tokens: int) -> None:
        self.caps.append(max_tokens)
        request.setdefault('parameters', {})['max_tokens'] = max_tokens


class _CapturingClient:

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []

    async def post(self, request: Dict[str, Any]) -> BenchmarkData:
        self.requests.append(dict(request))
        return BenchmarkData(
            request=json.dumps(request),
            start_time=0.0,
            completed_time=0.0,
            query_latency=0.0,
            first_chunk_latency=0.0,
            success=True,
            is_stream=True,
            prompt_tokens=1,
            completion_tokens=1,
            generated_text='ok',
        )


def test_worker_routes_per_turn_cap_through_hook() -> None:
    args = Arguments(model='m', api='openai', number=1, parallel=1, rate=-1, warmup_num=0, multi_turn=True)
    args.number = 1
    args.parallel = 1
    args.rate = -1
    conversations = [[Turn(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=37)]
                     for _ in range(args.total_count)]

    plugin = _CapPlugin()
    client = _CapturingClient()

    async def main() -> None:
        await MultiTurnStrategy(args, plugin, client, asyncio.Queue(), conversations).run()

    asyncio.run(main())

    assert client.requests, 'no request was dispatched'
    assert plugin.caps and all(cap == 37 for cap in plugin.caps)
    sent = client.requests[0]
    assert sent['parameters']['max_tokens'] == 37
    assert 'max_tokens' not in sent


if __name__ == '__main__':
    import unittest

    unittest.main(buffer=False)
