"""Multi-turn OpenAI chat completions through the bridge.

≥3 turns: assistant → tool_call → tool result → assistant. Verifies the
recorder bumps ``step`` correctly and parks TOOL_RESULT events on the
new step (matches the AgentLoop / Anthropic-bridge convention).
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


def _tool_output(call_id: str, name: str, args: dict) -> ModelOutput:
    tc = ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args), type='function')
    msg = ChatMessageAssistant(content='', tool_calls=[tc])
    return ModelOutput(model='mock', choices=[ChatCompletionChoice(message=msg, stop_reason='tool_calls')])


def _text_output(text: str) -> ModelOutput:
    msg = ChatMessageAssistant(content=text)
    return ModelOutput(model='mock', choices=[ChatCompletionChoice(message=msg, stop_reason='stop')])


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


def test_three_turn_tool_use_records_correct_step_layout():
    """T0 user→tool_call, T1 tool_result + user→tool_call, T2 tool_result→final."""
    outputs: List[ModelOutput] = [
        _tool_output('call-1', 'lookup', {'q': 'a'}),
        _tool_output('call-2', 'lookup', {'q': 'b'}),
        _text_output('done: a+b'),
    ]
    api = MockLLM(model_name='mock', custom_outputs=outputs)
    model = Model(api=api, config=GenerateConfig())

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/chat/completions'

            def _request(body: dict) -> dict:
                return _post_json(url, body, session.token)

            loop = asyncio.get_running_loop()
            # Turn 0
            messages = [{'role': 'user', 'content': 'look stuff up'}]
            r0 = await loop.run_in_executor(None, _request, {'model': 'mock', 'messages': list(messages)})
            assert r0['choices'][0]['finish_reason'] == 'tool_calls'
            assistant_0 = r0['choices'][0]['message']
            messages.append(assistant_0)
            messages.append({
                'role': 'tool',
                'tool_call_id': 'call-1',
                'content': 'result-a',
            })

            # Turn 1
            r1 = await loop.run_in_executor(None, _request, {'model': 'mock', 'messages': list(messages)})
            assistant_1 = r1['choices'][0]['message']
            messages.append(assistant_1)
            messages.append({
                'role': 'tool',
                'tool_call_id': 'call-2',
                'content': 'result-b',
            })

            # Turn 2
            r2 = await loop.run_in_executor(None, _request, {'model': 'mock', 'messages': list(messages)})
            assert r2['choices'][0]['message']['content'] == 'done: a+b'
            assert r2['choices'][0]['finish_reason'] == 'stop'

            return session.recorder.snapshot(), session.recorder.messages()

    trace, transcript = AsyncioLoopRunner.run(_go())

    # MODEL_GENERATE on steps 0, 1, 2
    gen_steps = sorted(ev.step for ev in trace.events if ev.type == EventType.MODEL_GENERATE)
    assert gen_steps == [0, 1, 2]

    # TOOL_CALL events on steps 0 and 1 (turn 2 has no tool_call)
    tool_call_steps = sorted(ev.step for ev in trace.events if ev.type == EventType.TOOL_CALL)
    assert tool_call_steps == [0, 1]

    # TOOL_RESULT lands on step+1 — for call-1 that's step 1, for call-2 step 2
    tool_result_events = [ev for ev in trace.events if ev.type == EventType.TOOL_RESULT]
    by_id = {ev.payload['id']: ev.step for ev in tool_result_events}
    assert by_id == {'call-1': 1, 'call-2': 2}

    # Reconstructed transcript should interleave user / assistant / tool / assistant / tool / assistant
    roles = [m.role for m in transcript]
    assert roles == ['user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']
    assert transcript[2].tool_call_id == 'call-1'
    assert transcript[4].tool_call_id == 'call-2'
    assert transcript[-1].text == 'done: a+b'
