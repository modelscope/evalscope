import asyncio
import json

from evalscope.agent.external.bridge.sse_anthropic import stream_anthropic_response
from evalscope.agent.external.bridge.translate_anthropic import model_output_to_anthropic_response
from evalscope.api.model import ModelOutput


def _tool_call_output(stop_reason):
    output = ModelOutput.for_tool_call('gemini', 'read_file', {'path': 'a.py'})
    output.choices[0].stop_reason = stop_reason
    return output


def test_anthropic_response_reports_tool_use_when_backend_stops_with_tool_calls():
    response = model_output_to_anthropic_response(_tool_call_output('stop'), request_model='claude')

    assert [block['type'] for block in response['content']] == ['text', 'tool_use']
    assert response['stop_reason'] == 'tool_use'


def test_anthropic_response_keeps_end_turn_without_tool_calls():
    output = ModelOutput.from_content('gemini', 'done')

    assert model_output_to_anthropic_response(output)['stop_reason'] == 'end_turn'


def test_anthropic_response_keeps_max_tokens_for_a_cut_off_tool_call():
    output = _tool_call_output('max_tokens')

    assert model_output_to_anthropic_response(output)['stop_reason'] == 'max_tokens'


def test_anthropic_stream_reports_tool_use_when_backend_stops_with_tool_calls():

    async def _go():
        task = asyncio.create_task(asyncio.sleep(0, result=_tool_call_output('stop')))
        deltas = []
        async for chunk in stream_anthropic_response(task, request_model='claude'):
            text = chunk.decode('utf-8')
            if text.startswith('event: message_delta'):
                deltas.append(json.loads(text.split('data: ', 1)[1].strip()))
        return deltas

    assert asyncio.run(_go())[0]['delta']['stop_reason'] == 'tool_use'
