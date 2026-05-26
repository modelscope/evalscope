"""Streaming function_call frames reassemble cleanly across parallel calls.

Codex parses ``response.function_call_arguments.delta`` frames incrementally
by ``item_id`` (Responses analogue of the chat completions ``index`` field).
Verify each delta carries its item_id and that delta chunks reassemble to
the full arguments JSON for every call.
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


def _multi_function_call_output() -> ModelOutput:
    """Two parallel function calls with non-trivial JSON arguments."""
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
        model='mock-responses',
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


def _parse_responses_sse(raw: str) -> list:
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


def test_parallel_function_calls_streamed_with_item_id_and_reassemble():
    """Each delta frame must carry ``item_id``; per-item reassembly matches the original arguments JSON."""

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        api = MockLLM(model_name='mock-responses', custom_outputs=[_multi_function_call_output()])
        model = Model(api=api, config=GenerateConfig())
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                return _post_stream(
                    url,
                    {
                        'model': 'mock-responses',
                        'input': [{
                            'type': 'message',
                            'role': 'user',
                            'content': [{'type': 'input_text', 'text': 'do two things'}],
                        }],
                        'stream': True,
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    events = _parse_responses_sse(raw)

    # Collect each function_call item's id (output_item.added carries the item dict).
    fc_item_ids: list = []
    for ev, data in events:
        if ev == 'response.output_item.added' and data['item'].get('type') == 'function_call':
            fc_item_ids.append(data['item']['id'])
    assert len(fc_item_ids) == 2, f'expected 2 function_call items, got {fc_item_ids!r}'

    # Reassemble arguments per item_id.
    reassembled: dict = defaultdict(str)
    for ev, data in events:
        if ev == 'response.function_call_arguments.delta':
            assert 'item_id' in data, 'every delta frame must carry item_id'
            reassembled[data['item_id']] += data['delta']

    # Match against the 'done' frames (which echo the full arguments string).
    done_args = {
        data['item_id']: data['arguments']
        for ev, data in events
        if ev == 'response.function_call_arguments.done'
    }
    assert set(done_args.keys()) == set(fc_item_ids)
    for item_id in fc_item_ids:
        assert reassembled[item_id] == done_args[item_id], (
            f'item {item_id}: reassembled={reassembled[item_id]!r} done={done_args[item_id]!r}'
        )

    # Each reassembled string is valid JSON matching the original args.
    args_by_call_id = {}
    for ev, data in events:
        if ev == 'response.output_item.done' and data['item'].get('type') == 'function_call':
            args_by_call_id[data['item']['call_id']] = json.loads(data['item']['arguments'])
    assert args_by_call_id['call-1'] == {'q': 'cats and dogs', 'limit': 5}
    assert args_by_call_id['call-2'] == {'text': 'hello world', 'lang': 'fr'}


def test_function_call_arguments_streamed_in_multiple_chunks():
    """Long arguments JSON arrives across ≥2 delta frames; concat reproduces it."""
    long_args = {'query': 'x' * 200, 'page': 1}
    tc = ToolCall(id='call-long', function=ToolFunction(name='lookup', arguments=long_args), type='function')
    msg = ChatMessageAssistant(content='', tool_calls=[tc])
    output = ModelOutput(model='mock-responses', choices=[ChatCompletionChoice(message=msg, stop_reason='tool_calls')])

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        api = MockLLM(model_name='mock-responses', custom_outputs=[output])
        model = Model(api=api, config=GenerateConfig())
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                return _post_stream(
                    url,
                    {
                        'model': 'mock-responses',
                        'input': [{
                            'type': 'message',
                            'role': 'user',
                            'content': [{'type': 'input_text', 'text': 'long call'}],
                        }],
                        'stream': True,
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    events = _parse_responses_sse(raw)
    deltas = [data['delta'] for ev, data in events if ev == 'response.function_call_arguments.delta']
    assert len(deltas) >= 2, f'expected ≥2 chunks for long arguments, got {len(deltas)}'
    reassembled = ''.join(deltas)
    assert json.loads(reassembled) == long_args
