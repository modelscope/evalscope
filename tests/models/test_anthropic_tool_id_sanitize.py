"""Tests for Anthropic tool call id sanitization (issue #1523).

Dataset-provided histories (e.g. general_fc) carry tool call ids like
``functions.search:0`` which violate Anthropic's ``^[a-zA-Z0-9_-]+$``
constraint and repeat across turns. The conversion layer must rewrite them
while preserving tool_use/tool_result pairing.
"""
import pytest
import re

from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.tool import ToolCall, ToolFunction

VALID_ID = re.compile(r'^[a-zA-Z0-9_-]+$')


def _anthropic_utils():
    pytest.importorskip('anthropic')
    from evalscope.models.utils import anthropic

    return anthropic


def _tool_call(call_id: str, name: str = 'search') -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments={'query': 'q'}))


def _collect_blocks(messages, block_type: str):
    blocks = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            continue
        blocks.extend(block for block in content if block.get('type') == block_type)
    return blocks


def test_illegal_and_duplicate_tool_ids_are_sanitized_and_paired():
    anthropic = _anthropic_utils()

    # Two turns reusing the same illegal id, mirroring general_fc data
    _, messages = anthropic.anthropic_chat_messages([
        ChatMessageUser(content='first question'),
        ChatMessageAssistant(content='', tool_calls=[_tool_call('functions.search:0')]),
        ChatMessageTool(content='result 1', tool_call_id='functions.search:0'),
        ChatMessageAssistant(content='', tool_calls=[_tool_call('functions.search:0')]),
        ChatMessageTool(content='result 2', tool_call_id='functions.search:0'),
        ChatMessageUser(content='second question'),
    ])

    tool_use_blocks = _collect_blocks(messages, 'tool_use')
    tool_result_blocks = _collect_blocks(messages, 'tool_result')
    assert len(tool_use_blocks) == 2
    assert len(tool_result_blocks) == 2

    tool_use_ids = [block['id'] for block in tool_use_blocks]
    tool_result_ids = [block['tool_use_id'] for block in tool_result_blocks]

    # All ids must satisfy Anthropic's constraint
    for tool_id in tool_use_ids + tool_result_ids:
        assert VALID_ID.match(tool_id), f'illegal id: {tool_id}'
    # Ids must be unique across turns
    assert len(set(tool_use_ids)) == 2
    # Each tool_result must pair with its own turn's tool_use
    assert tool_result_ids == tool_use_ids


def test_multiple_tool_calls_in_one_turn_keep_distinct_pairing():
    anthropic = _anthropic_utils()

    _, messages = anthropic.anthropic_chat_messages([
        ChatMessageUser(content='question'),
        ChatMessageAssistant(
            content='',
            tool_calls=[_tool_call('functions.search:0'), _tool_call('functions.search:1')],
        ),
        ChatMessageTool(content='result 0', tool_call_id='functions.search:0'),
        ChatMessageTool(content='result 1', tool_call_id='functions.search:1'),
        ChatMessageUser(content='follow up'),
    ])

    tool_use_ids = [block['id'] for block in _collect_blocks(messages, 'tool_use')]
    tool_result_ids = [block['tool_use_id'] for block in _collect_blocks(messages, 'tool_result')]

    assert len(set(tool_use_ids)) == 2
    assert tool_result_ids == tool_use_ids


def test_legal_tool_ids_are_kept_unchanged():
    anthropic = _anthropic_utils()

    _, messages = anthropic.anthropic_chat_messages([
        ChatMessageUser(content='question'),
        ChatMessageAssistant(content='', tool_calls=[_tool_call('toolu_abc123')]),
        ChatMessageTool(content='result', tool_call_id='toolu_abc123'),
        ChatMessageUser(content='follow up'),
    ])

    tool_use_ids = [block['id'] for block in _collect_blocks(messages, 'tool_use')]
    tool_result_ids = [block['tool_use_id'] for block in _collect_blocks(messages, 'tool_result')]

    assert tool_use_ids == ['toolu_abc123']
    assert tool_result_ids == ['toolu_abc123']


def test_sanitization_does_not_mutate_original_messages():
    anthropic = _anthropic_utils()

    assistant = ChatMessageAssistant(content='', tool_calls=[_tool_call('functions.search:0')])
    tool = ChatMessageTool(content='result', tool_call_id='functions.search:0')

    anthropic.anthropic_chat_messages([ChatMessageUser(content='q'), assistant, tool])

    assert assistant.tool_calls[0].id == 'functions.search:0'
    assert tool.tool_call_id == 'functions.search:0'
