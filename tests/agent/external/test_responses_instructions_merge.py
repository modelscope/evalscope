"""Top-level ``instructions`` + input[]-embedded system messages merge in order.

Responses API has two ways to push system context: a top-level
``instructions`` string and ``{type:'message', role:'system'|'developer',...}``
items in ``input[]``. The bridge merges both into the ``ChatMessage[]``
the model layer sees, preserving the spec'd order (instructions first,
then any input[] system items as they appear).

This test pokes the translator directly (``responses_request_to_messages``)
rather than going through the HTTP layer — the translator is the
canonical place where the merge is decided.
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
