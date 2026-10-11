"""Exercise the example's native agent dispatch with a deterministic HTTP MCP server."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

pytest.importorskip('mcp')

from evalscope.agent.runner import run_native_agent
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from examples.agent.parallel_search import build_task_config


def _output(name: str, arguments: dict[str, Any]) -> ModelOutput:
    return ModelOutput(choices=[
        ChatCompletionChoice(message=ChatMessageAssistant(
            content='', tool_calls=[ToolCall(id=name, function=ToolFunction(name=name, arguments=arguments))],
        )),
    ])


def test_parallel_example_native_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MODEL_NAME', 'example-model')
    monkeypatch.setenv('MODEL_API_URL', 'https://model.test/v1')
    monkeypatch.setenv('MODEL_API_KEY', 'model-secret')
    cfg = build_task_config()
    requests: list[httpx.Request] = []
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method != 'POST':
            return httpx.Response(405)
        payload = json.loads(request.content)
        method = payload['method']
        if 'id' not in payload:
            return httpx.Response(202)
        if method == 'initialize':
            result = {
                'protocolVersion': payload['params']['protocolVersion'],
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'parallel-fixture', 'version': '1.0'},
            }
        elif method == 'tools/list':
            result = {'tools': [
                {'name': name, 'inputSchema': {'type': 'object', 'properties': properties, 'required': required}}
                for name, properties, required in [
                    ('web_search', {'objective': {'type': 'string'},
                                    'search_queries': {'type': 'array', 'items': {'type': 'string'}}},
                     ['objective', 'search_queries']),
                    ('web_fetch', {'urls': {'type': 'array', 'items': {'type': 'string'}}}, ['urls']),
                    ('unused_tool', {}, []),
                ]
            ]}
        elif method == 'tools/call':
            name = payload['params']['name']
            calls.append(name)
            text = ('EvalScope docs: https://evalscope.readthedocs.io/en/latest/' if name == 'web_search'
                    else 'EvalScope supports native agent evaluation.')
            result = {'content': [{'type': 'text', 'text': text}], 'isError': False}
        else:
            raise AssertionError(f'Unexpected MCP method: {method}')
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': payload['id'], 'result': result})

    client_type = httpx.AsyncClient

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', make_client)
    model = MagicMock()
    model.generate_async = AsyncMock(side_effect=[
        _output('web_search', {'objective': 'Find EvalScope docs', 'search_queries': ['EvalScope native agent']}),
        _output('web_fetch', {'urls': ['https://evalscope.readthedocs.io/en/latest/']}),
        _output('submit', {'answer': 'EvalScope supports native agent evaluation.'}),
    ])
    try:
        result = run_native_agent(
            task_config=cfg,
            model=model,
            sample=Sample(input='What does EvalScope support?', target='native agent evaluation'),
            build_sandbox_config=lambda _: None,
            extract_final_answer=lambda loop_result, strategy: strategy.extract_final_answer(loop_result),
        )
        assert result.output.completion == 'EvalScope supports native agent evaluation.'
        observations = [message.text for message in result.messages if message.role == 'tool']
        assert any('https://evalscope.readthedocs.io/' in text for text in observations)
        assert any('native agent evaluation' in text for text in observations)
        assert calls == ['web_search', 'web_fetch']
        advertised = {tool.name for tool in model.generate_async.call_args_list[0].kwargs['tools']}
        assert {'web_search', 'web_fetch', 'submit'} <= advertised
        assert 'unused_tool' not in advertised
        assert requests
        for request in requests:
            assert str(request.url) == 'https://search.parallel.ai/mcp'
            assert request.headers['User-Agent'] == cfg.agent_config.mcp_servers[0].headers['User-Agent']
            assert 'Authorization' not in request.headers
            assert 'x-api-key' not in request.headers
            assert b'model-secret' not in request.content
        assert NativeAgentConfig().mcp_servers == []
    finally:
        AsyncioLoopRunner.shutdown_for_thread()
