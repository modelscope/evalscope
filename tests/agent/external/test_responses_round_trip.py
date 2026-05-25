"""Round-trip test for the OpenAI Responses bridge route (PR2).

Drives the bridge over a raw aiohttp client to stay independent of any
specific OpenAI SDK version. Covers JSON mode + SSE streaming, reasoning,
function_call output rendering, and the 401 unknown-token path.

The strict event-sequence properties of the SSE stream are covered by
``test_responses_event_sequence`` (which tests the synthesizer directly,
no HTTP). This file exercises the full handler chain end-to-end.
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
    api = MockLLM(model_name='mock-responses', custom_outputs=[output])
    return Model(api=api, config=GenerateConfig())


def _text_output(text: str) -> ModelOutput:
    msg = ChatMessageAssistant(content=text)
    return ModelOutput(model='mock-responses', choices=[ChatCompletionChoice(message=msg, stop_reason='stop')])


def _reasoning_output(reasoning: str, text: str) -> ModelOutput:
    msg = ChatMessageAssistant(content=[ContentReasoning(reasoning=reasoning), ContentText(text=text)])
    return ModelOutput(model='mock-responses', choices=[ChatCompletionChoice(message=msg, stop_reason='stop')])


def _function_call_output(name: str, args: dict, call_id: str = 'call-abc') -> ModelOutput:
    tc = ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args), type='function')
    msg = ChatMessageAssistant(content='', tool_calls=[tc])
    return ModelOutput(model='mock-responses', choices=[ChatCompletionChoice(message=msg, stop_reason='tool_calls')])


def _user_input_message(text: str) -> dict:
    return {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}


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


def _parse_responses_sse(raw: str) -> list:
    """Parse ``event: <type>\\ndata: <json>\\n\\n`` frames into ``(type, data)`` pairs."""
    out = []
    for chunk in raw.split('\n\n'):
        lines = chunk.strip().split('\n')
        if len(lines) < 2 or not lines[0].startswith('event: '):
            continue
        event = lines[0][len('event: '):]
        data_line = next((ln for ln in lines if ln.startswith('data: ')), None)
        if data_line is None:
            continue
        out.append((event, json.loads(data_line[len('data: '):])))
    return out


def test_responses_json_message_payload_shape():
    """JSON-mode response has correct top-level + output[message] item shape."""

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_text_output('hello from responses'))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                return _post_json(
                    url,
                    {
                        'model': 'mock-responses',
                        'input': [_user_input_message('hi')],
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    body = AsyncioLoopRunner.run(_go())
    assert body['object'] == 'response'
    assert body['status'] == 'completed'
    assert body['model'] == 'mock-responses'
    assert len(body['output']) == 1
    item = body['output'][0]
    assert item['type'] == 'message'
    assert item['role'] == 'assistant'
    assert item['content'][0]['type'] == 'output_text'
    assert item['content'][0]['text'] == 'hello from responses'
    assert set(body['usage']) == {'input_tokens', 'output_tokens', 'total_tokens'}


def test_responses_json_reasoning_renders_summary_item():
    """ContentReasoning block → output item with type=reasoning, summary_text."""

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_reasoning_output('thinking step by step', '42'))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                return _post_json(
                    url,
                    {
                        'model': 'mock-responses',
                        'input': [_user_input_message('why?')],
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    body = AsyncioLoopRunner.run(_go())
    types = [item['type'] for item in body['output']]
    assert types == ['reasoning', 'message']
    reasoning_item = body['output'][0]
    assert reasoning_item['summary'][0]['type'] == 'summary_text'
    assert reasoning_item['summary'][0]['text'] == 'thinking step by step'
    msg_item = body['output'][1]
    assert msg_item['content'][0]['text'] == '42'


def test_responses_json_function_call_uses_string_arguments():
    """function_call output item: arguments is a JSON-encoded string (per spec)."""

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_function_call_output('lookup', {'q': 'cats', 'limit': 3}))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                return _post_json(
                    url,
                    {
                        'model': 'mock-responses',
                        'input': [_user_input_message('find me cats')],
                        'tools': [{
                            'type': 'function',
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
                        }],
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    body = AsyncioLoopRunner.run(_go())
    fc_items = [it for it in body['output'] if it['type'] == 'function_call']
    assert len(fc_items) == 1
    fc = fc_items[0]
    assert fc['name'] == 'lookup'
    assert fc['call_id'] == 'call-abc'
    # arguments MUST be JSON-encoded string per Responses spec
    assert isinstance(fc['arguments'], str)
    assert json.loads(fc['arguments']) == {'q': 'cats', 'limit': 3}


def test_responses_streaming_text_round_trip_reassembles_via_output_text_deltas():
    """SSE deltas reassemble to the original text; required events present."""
    expected = 'streamed responses output: 42 ' * 4

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model(_text_output(expected))
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                return _post_stream(
                    url,
                    {
                        'model': 'mock-responses',
                        'input': [_user_input_message('stream?')],
                        'stream': True,
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    events = _parse_responses_sse(raw)
    event_types = [t for t, _ in events]

    # Spec bookends.
    assert event_types[0] == 'response.created'
    assert event_types[1] == 'response.in_progress'
    assert event_types[-1] == 'response.completed'
    # No [DONE] sentinel.
    assert '[DONE]' not in raw

    # Reassemble text from output_text.delta frames.
    reassembled = ''.join(
        data['delta'] for ev, data in events if ev == 'response.output_text.delta'
    )
    assert reassembled == expected

    # sequence_number is strictly monotonic.
    seqs = [data['sequence_number'] for _, data in events]
    assert seqs == list(range(1, len(seqs) + 1))


def test_responses_unknown_token_returns_401():
    """The bridge must 401 a Responses request with an unknown trial token."""

    async def _go() -> int:
        proxy = await ModelProxyServer.get_or_start()
        url = f'{proxy.base_url}/openai/v1/responses'

        def _request() -> int:
            req = urllib.request.Request(
                url,
                data=b'{"model":"x","input":[]}',
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
