"""Coverage for the extended input[] item types added in PR2 review:

* ``custom_tool_call`` + ``custom_tool_call_output`` — codex custom-tool
  variant; arguments are a free-form string wrapped as ``{'input': ...}``
* ``computer_call`` + ``computer_call_output`` — computer-use tool;
  output is typically ``{'image_url': ...}``
* ``web_search_call`` / ``mcp_call`` / ``file_search_call`` /
  ``code_interpreter_call`` — built-in tool uses rendered as opaque
  placeholder ContentText on the pending assistant message
* ``item_reference`` — stateful pointer, dropped with WARN
* unknown item types — WARN + skipped (forward-compat)

Verifies :mod:`translate_responses` directly; HTTP / SSE coverage stays
in the other test files.
"""

import logging
import pytest

from evalscope.agent.external.bridge.translate_responses import responses_request_to_messages
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ContentText


@pytest.fixture
def caplog_evalscope(caplog, monkeypatch):
    """Capture WARN+ from evalscope logger (which sets propagate=False by default)."""
    monkeypatch.setattr(logging.getLogger('evalscope'), 'propagate', True)
    caplog.set_level(logging.WARNING, logger='evalscope')
    return caplog


def test_custom_tool_call_and_output_round_trip():
    body = {
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'do stuff'}]},
            {
                'type': 'custom_tool_call',
                'call_id': 'c1',
                'name': 'shell',
                'input': 'ls -la',
            },
            {
                'type': 'custom_tool_call_output',
                'call_id': 'c1',
                'output': 'file1\nfile2',
            },
        ],
    }
    messages = responses_request_to_messages(body)
    roles = [m.role for m in messages]
    assert roles == ['user', 'assistant', 'tool']
    assistant = messages[1]
    assert isinstance(assistant, ChatMessageAssistant)
    assert assistant.tool_calls and len(assistant.tool_calls) == 1
    tc = assistant.tool_calls[0]
    assert tc.id == 'c1'
    assert tc.type == 'custom'
    assert tc.function.name == 'shell'
    assert tc.function.arguments == {'input': 'ls -la'}
    tool_msg = messages[2]
    assert isinstance(tool_msg, ChatMessageTool)
    assert tool_msg.tool_call_id == 'c1'
    assert tool_msg.text == 'file1\nfile2'


def test_computer_call_output_renders_image_placeholder():
    body = {
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'click'}]},
            {
                'type': 'computer_call',
                'call_id': 'cp1',
                'name': 'computer',
                'action': {'type': 'click', 'x': 100, 'y': 200},
            },
            {
                'type': 'computer_call_output',
                'call_id': 'cp1',
                'output': {'image_url': 'data:image/png;base64,iVBORw0KGgo='},
            },
        ],
    }
    messages = responses_request_to_messages(body)
    # user + assistant(computer_call rendered as placeholder text) + tool(image placeholder)
    assert [m.role for m in messages] == ['user', 'assistant', 'tool']
    assistant = messages[1]
    assert isinstance(assistant, ChatMessageAssistant)
    # No tool_calls — computer_call is a built-in, rendered as placeholder content.
    assert not assistant.tool_calls
    text = assistant.text
    assert '[computer_call' in text
    assert 'cp1' in text
    tool_msg = messages[2]
    assert isinstance(tool_msg, ChatMessageTool)
    assert tool_msg.tool_call_id == 'cp1'
    # tool_msg content rendered the dict; bridge cannot transmit binary images
    # to chat-only LLMs, so it gets a placeholder.
    assert '[image:' in tool_msg.text or 'image_url' in tool_msg.text


def test_builtin_tool_call_items_render_as_placeholder_text():
    body = {
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'find me cats'}]},
            {
                'type': 'web_search_call',
                'id': 'ws_001',
                'status': 'completed',
                'action': {'type': 'search', 'query': 'cat videos'},
            },
            {
                'type': 'mcp_call',
                'id': 'mc_001',
                'server_label': 'demo-server',
                'name': 'lookup',
                'output': 'mcp result text',
            },
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'thanks'}]},
        ],
    }
    messages = responses_request_to_messages(body)
    # user → assistant(placeholders for web_search + mcp) → user
    assert [m.role for m in messages] == ['user', 'assistant', 'user']
    assistant = messages[1]
    placeholders = [
        b.text for b in assistant.content
        if isinstance(b, ContentText)
    ] if isinstance(assistant.content, list) else [assistant.content]
    joined = '\n'.join(placeholders)
    assert '[web_search_call' in joined
    assert 'ws_001' in joined
    assert "'cat videos'" in joined or 'cat videos' in joined or "action='search'" in joined
    assert '[mcp_call' in joined
    assert 'mc_001' in joined
    assert "server_label='demo-server'" in joined
    assert "name='lookup'" in joined


def test_item_reference_logs_warning_and_drops(caplog_evalscope):
    body = {
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'continue'}]},
            {'type': 'item_reference', 'id': 'fc_stored_xyz'},
        ],
    }
    messages = responses_request_to_messages(body)
    # item_reference dropped → only the user message survives
    assert [m.role for m in messages] == ['user']
    warns = [r.getMessage() for r in caplog_evalscope.records if r.levelno >= logging.WARNING]
    assert any('item_reference' in w for w in warns), f'expected item_reference WARN; got {warns!r}'
    assert any('fc_stored_xyz' in w for w in warns)


def test_unknown_item_type_logs_warning_and_continues(caplog_evalscope):
    body = {
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]},
            {'type': 'totally_made_up_future_type', 'id': 'x_001'},
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'still here?'}]},
        ],
    }
    messages = responses_request_to_messages(body)
    assert [m.role for m in messages] == ['user', 'user']
    warns = [r.getMessage() for r in caplog_evalscope.records if r.levelno >= logging.WARNING]
    assert any('unsupported input item type' in w and 'totally_made_up_future_type' in w for w in warns)
