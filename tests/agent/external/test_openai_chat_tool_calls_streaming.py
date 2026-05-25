"""Streaming tool_call frames must include ``index`` and reassemble cleanly.

OpenAI clients fan out parallel tool_calls by the ``index`` field on each
delta entry — without it the SDK collapses everything into call 0 and the
arguments JSON gets corrupted.
"""

import asyncio
import json
import pytest
import urllib.request
from collections import defaultdict

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _multi_tool_output() -> ModelOutput:
    """Two parallel tool calls with non-trivial JSON arguments."""
    tcs = [
        ToolCall(
            id='call-1',
            function=ToolFunction(name='lookup', arguments={'q': 'cats and dogs', 'limit': 5}),
            type='function',
        ),
        ToolCall(
            id='call-2',
            function=ToolFunction(name='translate', arguments={'text': 'hello world', 'lang': 'fr'}),
            type='function',
        ),
    ]
    msg = ChatMessageAssistant(content='', tool_calls=tcs)
    return ModelOutput(
        model='mock-openai',
        choices=[ChatCompletionChoice(message=msg, stop_reason='tool_calls')],
    )


def _post_stream(url: str, body: dict, token: str) -> str:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
            'Accept': 'text/event-stream',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode('utf-8')


def _parse_sse(raw: str) -> list:
    out = []
    for chunk in raw.split('\n\n'):
        chunk = chunk.strip()
        if not chunk.startswith('data:'):
            continue
        payload = chunk[len('data:'):].strip()
        if payload == '[DONE]':
            out.append('[DONE]')
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return out


def test_parallel_tool_calls_streamed_with_index_and_reassemble():
    api = MockLLM(model_name='mock-openai', custom_outputs=[_multi_tool_output()])
    model = Model(api=api, config=GenerateConfig())

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request():
                return _post_stream(
                    url,
                    {
                        'model': 'mock-openai',
                        'messages': [{'role': 'user', 'content': 'do two things'}],
                        'stream': True,
                        'parallel_tool_calls': True,
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    events = _parse_sse(raw)
    assert events[-1] == '[DONE]'

    # collected by index → (id, name, accumulated arguments string)
    ids_by_index: dict = {}
    names_by_index: dict = {}
    args_by_index: dict = defaultdict(str)
    finish = None
    for ev in events[:-1]:
        choice = ev['choices'][0]
        delta = choice.get('delta', {})
        for tc in delta.get('tool_calls') or []:
            assert 'index' in tc, 'every tool_call delta entry must carry index'
            idx = tc['index']
            if 'id' in tc:
                ids_by_index[idx] = tc['id']
            fn = tc.get('function') or {}
            if 'name' in fn:
                names_by_index[idx] = fn['name']
            if 'arguments' in fn:
                args_by_index[idx] += fn['arguments']
        if choice.get('finish_reason'):
            finish = choice['finish_reason']

    assert finish == 'tool_calls'
    assert ids_by_index == {0: 'call-1', 1: 'call-2'}
    assert names_by_index == {0: 'lookup', 1: 'translate'}
    # Args must reassemble to valid JSON matching the originals
    assert json.loads(args_by_index[0]) == {'q': 'cats and dogs', 'limit': 5}
    assert json.loads(args_by_index[1]) == {'text': 'hello world', 'lang': 'fr'}


def test_tool_call_arguments_streamed_in_multiple_chunks():
    """The arguments JSON for a single call is split across several deltas."""
    # Use a longer JSON string to force chunking (sse_openai chunks at 20 chars)
    big_args = {'paragraphs': ['lorem ipsum dolor sit amet'] * 4, 'count': 4}
    tc = ToolCall(id='call-X', function=ToolFunction(name='write', arguments=big_args), type='function')
    msg = ChatMessageAssistant(content='', tool_calls=[tc])
    output = ModelOutput(model='mock-openai', choices=[ChatCompletionChoice(message=msg, stop_reason='tool_calls')])
    api = MockLLM(model_name='mock-openai', custom_outputs=[output])
    model = Model(api=api, config=GenerateConfig())

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request():
                return _post_stream(
                    url,
                    {
                        'model': 'mock-openai',
                        'messages': [{'role': 'user', 'content': 'do it'}],
                        'stream': True,
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    events = _parse_sse(raw)
    arg_deltas = 0
    accumulated = ''
    for ev in events[:-1]:
        for tc in (ev['choices'][0].get('delta', {}).get('tool_calls') or []):
            fn = tc.get('function') or {}
            if 'arguments' in fn and fn['arguments']:
                # the announce frame has arguments='' so we skip empties
                arg_deltas += 1
                accumulated += fn['arguments']
    assert arg_deltas >= 2, 'large arguments JSON should split into multiple delta frames'
    assert json.loads(accumulated) == big_args
