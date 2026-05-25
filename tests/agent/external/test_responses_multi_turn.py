"""Multi-turn Responses round-trip through the bridge.

≥3 turns: assistant function_call → function_call_output → assistant
continues. Verifies the recorder bumps ``step`` correctly across the
Responses ``input[]`` shape (not chat's ``messages[]``) and that
TOOL_RESULT events park on the new step (matches the chat / anthropic
convention).
"""

import asyncio
import json
import pytest
import urllib.request
from typing import List

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.agent import EventType
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


def _function_call_output(call_id: str, name: str, args: dict) -> ModelOutput:
    tc = ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args), type='function')
    msg = ChatMessageAssistant(content='', tool_calls=[tc])
    return ModelOutput(model='mock-responses', choices=[ChatCompletionChoice(message=msg, stop_reason='tool_calls')])


def _text_output(text: str) -> ModelOutput:
    msg = ChatMessageAssistant(content=text)
    return ModelOutput(model='mock-responses', choices=[ChatCompletionChoice(message=msg, stop_reason='stop')])


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


def test_three_turn_responses_tool_use_records_correct_step_layout():
    """T0 user→function_call, T1 function_call_output + continue→function_call, T2 function_call_output→final."""
    outputs: List[ModelOutput] = [
        _function_call_output('call-1', 'lookup', {'q': 'a'}),
        _function_call_output('call-2', 'lookup', {'q': 'b'}),
        _text_output('done: a+b'),
    ]
    api = MockLLM(model_name='mock-responses', custom_outputs=outputs)
    model = Model(api=api, config=GenerateConfig())

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request(body: dict) -> dict:
                return _post_json(url, body, session.token)

            loop = asyncio.get_running_loop()

            # Turn 0 — user message only
            input_items: List[dict] = [{
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': 'look stuff up'}],
            }]
            r0 = await loop.run_in_executor(None, _request, {'model': 'mock-responses', 'input': list(input_items)})
            fc_items_0 = [it for it in r0['output'] if it['type'] == 'function_call']
            assert fc_items_0, f'turn 0 missing function_call output: {r0}'
            # Append the function_call item + its output to the running input for turn 1.
            input_items.append(fc_items_0[0])
            input_items.append({
                'type': 'function_call_output',
                'call_id': 'call-1',
                'output': 'result-a',
            })

            # Turn 1
            r1 = await loop.run_in_executor(None, _request, {'model': 'mock-responses', 'input': list(input_items)})
            fc_items_1 = [it for it in r1['output'] if it['type'] == 'function_call']
            assert fc_items_1
            input_items.append(fc_items_1[0])
            input_items.append({
                'type': 'function_call_output',
                'call_id': 'call-2',
                'output': 'result-b',
            })

            # Turn 2
            r2 = await loop.run_in_executor(None, _request, {'model': 'mock-responses', 'input': list(input_items)})
            msg_items = [it for it in r2['output'] if it['type'] == 'message']
            assert msg_items[0]['content'][0]['text'] == 'done: a+b'
            assert all(it['type'] != 'function_call' for it in r2['output'])

            return session.recorder.snapshot(), session.recorder.messages()

    trace, transcript = AsyncioLoopRunner.run(_go())

    # MODEL_GENERATE on steps 0, 1, 2
    gen_steps = sorted(ev.step for ev in trace.events if ev.type == EventType.MODEL_GENERATE)
    assert gen_steps == [0, 1, 2]

    # TOOL_CALL events on steps 0 and 1 (turn 2 has no tool_call)
    tool_call_steps = sorted(ev.step for ev in trace.events if ev.type == EventType.TOOL_CALL)
    assert tool_call_steps == [0, 1]

    # TOOL_RESULT lands on step+1
    tool_result_events = [ev for ev in trace.events if ev.type == EventType.TOOL_RESULT]
    by_id = {ev.payload['id']: ev.step for ev in tool_result_events}
    assert by_id == {'call-1': 1, 'call-2': 2}

    # Reconstructed transcript interleaves user / assistant / tool / assistant / tool / assistant
    roles = [m.role for m in transcript]
    assert roles == ['user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']
    assert transcript[2].tool_call_id == 'call-1'
    assert transcript[4].tool_call_id == 'call-2'
    assert transcript[-1].text == 'done: a+b'
