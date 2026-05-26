"""Round-trip test for the OpenAI Chat Completions bridge route.

Drives the bridge with a raw aiohttp client so the test stays independent
of any specific OpenAI SDK version. Covers JSON mode, SSE streaming, the
``reasoning_content`` vendor extension, and tool_calls in the response.
"""

import asyncio
import json
import pytest
import urllib.error
import urllib.request

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.messages import ChatMessageAssistant, ContentReasoning, ContentText
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_model(output: ModelOutput) -> Model:
    api = MockLLM(model_name='mock-openai', custom_outputs=[output])
    return Model(api=api, config=GenerateConfig())


def _text_output(text: str) -> ModelOutput:
    msg = ChatMessageAssistant(content=text)
    return ModelOutput(
        model='mock-openai',
        choices=[ChatCompletionChoice(message=msg, stop_reason='stop')],
    )


def _reasoning_output(reasoning: str, text: str) -> ModelOutput:
    msg = ChatMessageAssistant(content=[ContentReasoning(reasoning=reasoning), ContentText(text=text)])
    return ModelOutput(
        model='mock-openai',
        choices=[ChatCompletionChoice(message=msg, stop_reason='stop')],
    )


def _tool_call_output(name: str, args: dict) -> ModelOutput:
    tc = ToolCall(id='call-abc', function=ToolFunction(name=name, arguments=args), type='function')
    msg = ChatMessageAssistant(content='', tool_calls=[tc])
    return ModelOutput(
        model='mock-openai',
        choices=[ChatCompletionChoice(message=msg, stop_reason='tool_calls')],
    )


def _post_json(url: str, body: dict, token: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))


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


def _parse_sse_events(raw: str) -> list:
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


def test_openai_chat_completion_json_response():
    """JSON-mode response contains the text + correct finish_reason + usage shape."""
    expected = 'hello from mock'

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_text_output(expected))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request():
                return _post_json(
                    url,
                    {
                        'model': 'mock-openai',
                        'messages': [{'role': 'user', 'content': 'hi'}],
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    body = AsyncioLoopRunner.run(_go())
    assert body['object'] == 'chat.completion'
    assert body['choices'][0]['message']['content'] == expected
    assert body['choices'][0]['message']['role'] == 'assistant'
    assert body['choices'][0]['finish_reason'] == 'stop'
    assert 'usage' in body
    assert set(body['usage']) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}


def test_openai_chat_completion_reasoning_content_json():
    """reasoning_content from a ContentReasoning block appears alongside content."""

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_reasoning_output('let me think...', 'the answer is 42'))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request():
                return _post_json(
                    url,
                    {
                        'model': 'mock-openai',
                        'messages': [{'role': 'user', 'content': 'why?'}],
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    body = AsyncioLoopRunner.run(_go())
    msg = body['choices'][0]['message']
    assert msg['content'] == 'the answer is 42'
    assert msg['reasoning_content'] == 'let me think...'


def test_openai_chat_completion_tool_calls_json():
    """assistant tool_calls are rendered with JSON-string arguments + tool_calls finish_reason."""

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_tool_call_output('lookup', {'q': 'cats', 'limit': 3}))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request():
                return _post_json(
                    url,
                    {
                        'model': 'mock-openai',
                        'messages': [{'role': 'user', 'content': 'find me cats'}],
                        'tools': [{
                            'type': 'function',
                            'function': {
                                'name': 'lookup',
                                'description': 'find things',
                                'parameters': {
                                    'type': 'object',
                                    'properties': {
                                        'q': {'type': 'string'},
                                        'limit': {'type': 'integer'},
                                    },
                                    'required': ['q'],
                                },
                            },
                        }],
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    body = AsyncioLoopRunner.run(_go())
    msg = body['choices'][0]['message']
    assert body['choices'][0]['finish_reason'] == 'tool_calls'
    assert len(msg['tool_calls']) == 1
    tc = msg['tool_calls'][0]
    assert tc['id'] == 'call-abc'
    assert tc['type'] == 'function'
    assert tc['function']['name'] == 'lookup'
    # arguments MUST be a JSON-encoded string per OpenAI spec
    assert isinstance(tc['function']['arguments'], str)
    parsed_args = json.loads(tc['function']['arguments'])
    assert parsed_args == {'q': 'cats', 'limit': 3}


def test_openai_chat_completion_streaming_text_round_trip():
    """SSE stream reassembles to original text via content deltas."""
    expected = 'streamed answer: 42 ' * 4

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_text_output(expected))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request():
                return _post_stream(
                    url,
                    {
                        'model': 'mock-openai',
                        'messages': [{'role': 'user', 'content': 'stream?'}],
                        'stream': True,
                        'stream_options': {'include_usage': True},
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    events = _parse_sse_events(raw)
    assert events[-1] == '[DONE]'

    role_seen = False
    collected = ''
    finish = None
    usage = None
    for ev in events[:-1]:
        choice = ev['choices'][0]
        delta = choice.get('delta', {})
        if delta.get('role') == 'assistant':
            role_seen = True
        if 'content' in delta and delta['content']:
            collected += delta['content']
        if choice.get('finish_reason'):
            finish = choice['finish_reason']
            usage = ev.get('usage')
    assert role_seen, 'first SSE frame must announce role=assistant'
    assert collected == expected
    assert finish == 'stop'
    assert usage is not None
    assert set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}


def test_openai_chat_completion_streaming_reasoning_delta():
    """reasoning_content surfaces as delta.reasoning_content frames."""

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_reasoning_output('thinking out loud', 'final answer'))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request():
                return _post_stream(
                    url,
                    {
                        'model': 'mock-openai',
                        'messages': [{'role': 'user', 'content': '?'}],
                        'stream': True,
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    events = _parse_sse_events(raw)
    reasoning_collected = ''
    text_collected = ''
    for ev in events[:-1]:
        delta = ev['choices'][0].get('delta', {})
        if 'reasoning_content' in delta:
            reasoning_collected += delta['reasoning_content']
        if delta.get('content'):
            text_collected += delta['content']
    assert reasoning_collected == 'thinking out loud'
    assert text_collected == 'final answer'


def test_openai_chat_completion_unknown_token_returns_401():
    """The bridge must 401 requests bearing an unknown trial token."""

    async def _go() -> int:
        proxy = await ModelProxyServer.get_or_start()
        url = f'{proxy.base_url}/openai/v1/chat/completions'

        def _request() -> int:
            req = urllib.request.Request(
                url,
                data=b'{"model":"x","messages":[]}',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer trial-bogus',
                },
                method='POST',
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status
            except urllib.error.HTTPError as exc:
                return exc.code

        return await asyncio.get_running_loop().run_in_executor(None, _request)

    status = AsyncioLoopRunner.run(_go())
    assert status == 401
