"""Tests for reading tool calls out of a local model's text output (issue #1908).

A local checkpoint returns the chat template's ``<tool_call>`` blocks as plain
text. Before this parser existed the ModelScope generate path ignored them, so
every tool-calling benchmark scored the model on an empty answer.
"""
import logging

import pytest

import evalscope.models.utils.tool_call_text as tool_call_text
from evalscope.models.utils.tool_call_text import parse_tool_calls_from_text

CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'


def test_single_call_keeps_surrounding_text():
    content, calls = parse_tool_calls_from_text(f'Let me check the weather.\n{CALL}')

    assert content == 'Let me check the weather.'
    assert calls is not None and len(calls) == 1
    assert calls[0].function.name == 'get_weather'
    assert calls[0].function.arguments == {'city': 'Beijing'}


def test_call_only_leaves_empty_content():
    content, calls = parse_tool_calls_from_text(CALL)

    assert content == ''
    assert calls is not None and len(calls) == 1


def test_several_calls_keep_their_order():
    second = '<tool_call>{"name": "get_time", "arguments": {"zone": "UTC"}}</tool_call>'
    _, calls = parse_tool_calls_from_text(f'{CALL}\n{second}')

    assert calls is not None
    assert [call.function.name for call in calls] == ['get_weather', 'get_time']
    assert len({call.id for call in calls}) == 2


def test_nested_function_payload_is_accepted():
    block = '<tool_call>{"function": {"name": "search", "arguments": {"q": "x"}}}</tool_call>'
    _, calls = parse_tool_calls_from_text(block)

    assert calls is not None
    assert calls[0].function.name == 'search'
    assert calls[0].function.arguments == {'q': 'x'}


def test_arguments_written_as_a_json_string_are_parsed():
    block = '<tool_call>{"name": "ping", "arguments": "{\\"host\\": \\"example.com\\"}"}</tool_call>'
    _, calls = parse_tool_calls_from_text(block)

    assert calls is not None
    assert calls[0].function.arguments == {'host': 'example.com'}


def test_malformed_block_is_left_in_the_text():
    output = 'before <tool_call>{not json}</tool_call> after'
    content, calls = parse_tool_calls_from_text(output)

    assert calls is None
    assert content == output


def test_output_without_blocks_is_untouched():
    output = 'plain answer'
    content, calls = parse_tool_calls_from_text(output)

    assert calls is None
    assert content == output


def test_empty_output():
    assert parse_tool_calls_from_text('') == ('', None)


def test_block_without_a_name_is_not_a_call():
    output = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
    content, calls = parse_tool_calls_from_text(output)

    assert calls is None
    assert content == output


block = '<tool_call>'
close = '</tool_call>'


@pytest.fixture(autouse=True)
def reset_diagnostic():
    tool_call_text._DIAGNOSTIC_EMITTED = False
    yield
    tool_call_text._DIAGNOSTIC_EMITTED = False

PARAMS_CALL = f'{block}\n{{"name": "get_weather", "parameters": {{"city": "Beijing"}}}}\n{close}'

MISTRAL_ARRAY = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'



def test_parameters_key_is_accepted_as_an_alias():
    _, calls = parse_tool_calls_from_text(PARAMS_CALL)

    assert calls is not None
    assert calls[0].function.name == 'get_weather'
    assert calls[0].function.arguments == {'city': 'Beijing'}


def test_nested_function_payload_with_parameters_key():
    call = f'{block}{{"function": {{"name": "search", "parameters": {{"q": "x"}}}}}}{close}'
    _, calls = parse_tool_calls_from_text(call)

    assert calls is not None
    assert calls[0].function.arguments == {'q': 'x'}


def test_mistral_style_array_is_parsed():
    content, calls = parse_tool_calls_from_text(MISTRAL_ARRAY)

    assert content == ''
    assert calls is not None and len(calls) == 1
    assert calls[0].function.name == 'get_weather'
    assert calls[0].function.arguments == {'city': 'Beijing'}


def test_mistral_array_with_several_calls_keeps_order():
    output = '[TOOL_CALLS][{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]'
    _, calls = parse_tool_calls_from_text(output)

    assert calls is not None
    assert [call.function.name for call in calls] == ['a', 'b']
    assert calls[0].function.arguments == {'x': 1}
    assert calls[1].function.arguments == {'y': 2}


def test_unclosed_block_emits_a_diagnostic_and_stays_plain_text(caplog, reset_diagnostic):
    output = block + '{"name": "get_weather"}'

    with caplog.at_level(logging.WARNING):
        content, calls = parse_tool_calls_from_text(output)

    assert calls is None
    assert content == output
    assert any('tool call' in record.message.lower() for record in caplog.records)


def test_plain_text_answer_emits_no_diagnostic(caplog):
    with caplog.at_level(logging.WARNING):
        content, calls = parse_tool_calls_from_text('I can help with that.')

    assert calls is None
    assert content == 'I can help with that.'
    assert caplog.records == []


def test_diagnostic_is_logged_only_once(reset_diagnostic, caplog):
    with caplog.at_level(logging.WARNING):
        parse_tool_calls_from_text(block + '{"name": "a"}')
        parse_tool_calls_from_text('[TOOL_CALLS] [{"name": "b"}')

    warnings = [record for record in caplog.records if 'tool call' in record.message.lower()]
    assert len(warnings) == 1


def test_mixed_formats_keep_source_order_and_clean_text():
    array = '[TOOL_CALLS] [{"name": "first", "arguments": {}}]'
    block_call = f'{block}{{"name": "second", "arguments": {{}}}}{close}'
    content, calls = parse_tool_calls_from_text(array + '\n' + block_call)

    assert [call.function.name for call in calls] == ['first', 'second']
    assert content == ''


def test_partially_invalid_array_is_kept_as_text():
    output = '[TOOL_CALLS] [{"name": "valid", "arguments": {}}, "broken"]'
    content, calls = parse_tool_calls_from_text(output)

    assert calls is None
    assert content == output


def test_consecutive_arrays_and_trailing_text():
    output = (
        '[TOOL_CALLS] [{"name": "a", "arguments": {"x": 1}}]'
        ' [TOOL_CALLS] [{"name": "b", "arguments": {"y": 2}}] done'
    )
    content, calls = parse_tool_calls_from_text(output)

    assert [call.function.name for call in calls] == ['a', 'b']
    assert calls[0].function.arguments == {'x': 1}
    assert calls[1].function.arguments == {'y': 2}
    assert content == 'done'
