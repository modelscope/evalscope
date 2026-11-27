import asyncio
import json

import pytest

from evalscope.agent.external.bridge.sse_openai import stream_openai_response
from evalscope.agent.external.bridge.translate_openai import model_output_to_openai_response
from evalscope.api.model import ModelOutput


def _tool_call_output(stop_reason):
    output = ModelOutput.for_tool_call('gemini', 'read_file', {'path': 'a.py'})
    output.choices[0].stop_reason = stop_reason
    return output


def test_openai_response_reports_tool_calls_when_backend_stops_with_tool_calls():
    response = model_output_to_openai_response(_tool_call_output('stop'), request_model='gpt')

    assert response['choices'][0]['message']['tool_calls']
    assert response['choices'][0]['finish_reason'] == 'tool_calls'


def test_openai_response_keeps_stop_without_tool_calls():
    output = ModelOutput.from_content('gemini', 'done')

    assert model_output_to_openai_response(output)['choices'][0]['finish_reason'] == 'stop'


def test_openai_response_keeps_length_for_a_cut_off_tool_call():
    response = model_output_to_openai_response(_tool_call_output('max_tokens'))

    assert response['choices'][0]['finish_reason'] == 'length'


def _stream_finish_reasons(stop_reason):

    async def _go():
        task = asyncio.create_task(asyncio.sleep(0, result=_tool_call_output(stop_reason)))
        reasons = []
        async for chunk in stream_openai_response(task, request_model='gpt'):
            text = chunk.decode('utf-8').strip()
            if not text.startswith('data: ') or text == 'data: [DONE]':
                continue
            reason = json.loads(text[len('data: '):])['choices'][0]['finish_reason']
            if reason is not None:
                reasons.append(reason)
        return reasons

    return asyncio.run(_go())


def test_openai_stream_reports_tool_calls_when_backend_stops_with_tool_calls():
    assert _stream_finish_reasons('stop') == ['tool_calls']


def test_openai_stream_keeps_length_for_a_cut_off_tool_call():
    assert _stream_finish_reasons('max_tokens') == ['length']
