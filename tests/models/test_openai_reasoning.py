"""Unit tests for the three reasoning_format modes in openai_chat_message.

Each test pins the outgoing assistant-message dict shape so the breakage of
DeepSeek V4 thinking (issue #1392) cannot regress silently. The matrix covers
the three modes against representative ChatMessageAssistant shapes
(reasoning + text, tool_calls + reasoning, plain text, multiple reasoning
parts, blank reasoning).
"""

import pytest
from pydantic import ValidationError

from evalscope.api.messages import ChatMessageAssistant, ContentReasoning, ContentText
from evalscope.api.model import GenerateConfig
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.models.utils.openai import openai_chat_message


def _asst_reasoning_and_text(reasoning: str, text: str) -> ChatMessageAssistant:
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning), ContentText(text=text)],
    )


def _asst_tool_call_with_reasoning(reasoning: str) -> ChatMessageAssistant:
    tc = ToolCall(
        id='call-1',
        function=ToolFunction(name='lookup', arguments={'q': 'paris'}),
        type='function',
    )
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning)],
        tool_calls=[tc],
    )


# ---------------------------------------------------------------------------
# think_tag (default / pre-fix behavior)
# ---------------------------------------------------------------------------


def test_think_tag_embeds_reasoning_in_content():
    msg = _asst_reasoning_and_text('user wants paris weather', 'It is sunny.')
    out = openai_chat_message(msg, reasoning_format='think_tag')
    assert out['role'] == 'assistant'
    assert '<think>' in out['content']
    assert 'user wants paris weather' in out['content']
    assert 'It is sunny.' in out['content']
    assert 'reasoning_content' not in out


def test_think_tag_is_the_default():
    msg = _asst_reasoning_and_text('cot', 'reply')
    out = openai_chat_message(msg)
    assert '<think>' in out['content']
    assert 'reasoning_content' not in out


# ---------------------------------------------------------------------------
# reasoning_field (the fix)
# ---------------------------------------------------------------------------


def test_reasoning_field_promotes_to_top_level():
    msg = _asst_reasoning_and_text('user wants paris weather', 'It is sunny.')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert '<think>' not in out['content']
    assert 'It is sunny.' in out['content']
    assert out['reasoning_content'] == 'user wants paris weather'


def test_reasoning_field_with_tool_calls():
    msg = _asst_tool_call_with_reasoning('need to call lookup')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert out['reasoning_content'] == 'need to call lookup'
    assert '<think>' not in out['content']
    assert len(out['tool_calls']) == 1
    assert out['tool_calls'][0]['function']['name'] == 'lookup'


def test_reasoning_field_picks_last_when_multiple():
    msg = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning='first thought'),
            ContentText(text='intermediate'),
            ContentReasoning(reasoning='final thought'),
            ContentText(text='reply'),
        ],
    )
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert out['reasoning_content'] == 'final thought'


def test_reasoning_field_blank_falls_back_to_single_space():
    msg = _asst_reasoning_and_text('   ', 'reply')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert out['reasoning_content'] == ' '


def test_reasoning_field_no_reasoning_omits_key():
    msg = ChatMessageAssistant(content='plain reply, no reasoning')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert 'reasoning_content' not in out
    assert out['content'] == 'plain reply, no reasoning'


# ---------------------------------------------------------------------------
# none (drop reasoning entirely, for DeepSeek R1 etc.)
# ---------------------------------------------------------------------------


def test_none_drops_reasoning_entirely():
    msg = _asst_reasoning_and_text('cot text', 'reply text')
    out = openai_chat_message(msg, reasoning_format='none')
    assert '<think>' not in out['content']
    assert 'cot text' not in out['content']
    assert 'reply text' in out['content']
    assert 'reasoning_content' not in out


def test_none_keeps_tool_calls():
    msg = _asst_tool_call_with_reasoning('cot')
    out = openai_chat_message(msg, reasoning_format='none')
    assert 'reasoning_content' not in out
    assert '<think>' not in out['content']
    assert len(out['tool_calls']) == 1


# ---------------------------------------------------------------------------
# GenerateConfig schema invariants (breaking change verification)
# ---------------------------------------------------------------------------


def test_reasoning_history_accepts_new_literal_values():
    for v in ('none', 'think_tag', 'reasoning_field'):
        cfg = GenerateConfig(reasoning_history=v)
        assert cfg.reasoning_history == v


def test_reasoning_history_rejects_legacy_literal_values():
    for v in ('all', 'last', 'auto'):
        with pytest.raises(ValidationError):
            GenerateConfig(reasoning_history=v)


def test_reasoning_history_default_is_none():
    assert GenerateConfig().reasoning_history is None
