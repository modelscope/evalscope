"""``input[]`` walker correctness — ordering and content-block assembly.

Covers the cross-item invariants the translator must respect:
* Top-level ``instructions`` becomes a system message and lands BEFORE
  any user message
* ``input[]``-embedded ``role:'system'|'developer'`` items follow
  ``instructions`` in document order
* A ``reasoning`` item immediately followed by an assistant ``message``
  belongs to the same turn — the resulting assistant message must hold
  ``ContentReasoning`` then ``ContentText`` in that order

These pokes the translator (:func:`responses_request_to_messages`)
directly rather than going through the HTTP layer — the translator is
the canonical place where ordering is decided. Per-item-type semantics
(custom/computer/web_search/mcp/...) live in
``test_responses_extended_item_types.py``.
"""

from evalscope.agent.external.bridge.translate_responses import responses_request_to_messages


def test_top_level_instructions_emitted_before_first_user():
    """``instructions`` becomes a ChatMessageSystem placed before the user."""
    body = {
        'instructions': 'You are a helpful agent.',
        'input': [{
            'type': 'message',
            'role': 'user',
            'content': [{'type': 'input_text', 'text': 'hi'}],
        }],
    }
    messages = responses_request_to_messages(body)
    roles = [m.role for m in messages]
    assert roles == ['system', 'user']
    assert messages[0].text == 'You are a helpful agent.'
    assert messages[1].text == 'hi'


def test_input_system_appended_after_instructions_in_input_order():
    """``instructions`` first, then input[] system items in document order."""
    body = {
        'instructions': 'Top instructions.',
        'input': [
            {
                'type': 'message',
                'role': 'system',
                'content': [{'type': 'input_text', 'text': 'embedded system 1'}],
            },
            {
                'type': 'message',
                'role': 'developer',
                'content': [{'type': 'input_text', 'text': 'embedded developer 2'}],
            },
            {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': 'user prompt'}],
            },
        ],
    }
    messages = responses_request_to_messages(body)
    roles = [m.role for m in messages]
    assert roles == ['system', 'system', 'system', 'user']
    assert messages[0].text == 'Top instructions.'
    assert messages[1].text == 'embedded system 1'
    assert messages[2].text == 'embedded developer 2'  # developer treated as system
    assert messages[3].text == 'user prompt'


def test_no_instructions_only_input_system_works():
    """Bare input[] with a system message still produces system + user."""
    body = {
        'input': [
            {
                'type': 'message',
                'role': 'system',
                'content': [{'type': 'input_text', 'text': 'only input system'}],
            },
            {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': 'ask'}],
            },
        ],
    }
    messages = responses_request_to_messages(body)
    roles = [m.role for m in messages]
    assert roles == ['system', 'user']
    assert messages[0].text == 'only input system'


def test_reasoning_then_message_keeps_reasoning_in_front_of_text():
    """A ``reasoning`` item immediately followed by an assistant ``message``
    must collapse into a single assistant turn with the ContentReasoning
    block leading the ContentText (chain-of-thought → answer ordering)."""
    body = {
        'input': [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'why?'}]},
            {
                'type': 'reasoning',
                'summary': [{'type': 'summary_text', 'text': 'because of physics'}],
            },
            {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'the answer is 42'}],
            },
        ],
    }
    messages = responses_request_to_messages(body)
    assert [m.role for m in messages] == ['user', 'assistant']
    assistant = messages[1]
    assert isinstance(assistant.content, list)
    types = [type(b).__name__ for b in assistant.content]
    assert types == ['ContentReasoning', 'ContentText']
    assert assistant.content[0].reasoning == 'because of physics'
    assert assistant.content[1].text == 'the answer is 42'
