"""Tests for reading tool calls out of a local model's text output (issue #1908).

A local checkpoint returns the chat template's ``<tool_call>`` blocks as plain
text. Before this parser existed the ModelScope generate path ignored them, so
every tool-calling benchmark scored the model on an empty answer.
"""
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
