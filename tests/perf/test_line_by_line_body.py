"""Tests for line_by_line complete-body (JSON object) handling in OpenaiPlugin.

A complete request body (a ``dict``) must honor its own fields; CLI-level
generation parameters only fill in fields that are missing (``setdefault``
semantics). This prevents CLI defaults (``max_tokens=2048``, ``temperature=0.0``,
``stream=True``) from silently overwriting a user-supplied body.
"""

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin


def _make_plugin(**kwargs) -> OpenaiPlugin:
    args = Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        **kwargs,
    )
    return OpenaiPlugin(args)


def _messages():
    return [{'role': 'user', 'content': 'hi'}]


# ---------------------------------------------------------------------------
# Complete body: fill semantics
# ---------------------------------------------------------------------------

class TestCompleteBodyFillSemantics:
    """A dict body preserves its own fields; CLI only fills missing ones."""

    def test_preserves_body_generation_params(self):
        plugin = _make_plugin()  # CLI defaults: max_tokens=2048, temperature=0.0
        body = {'messages': _messages(), 'temperature': 0.6, 'max_tokens': 128}
        result = plugin.build_request(body)
        assert result['temperature'] == 0.6
        assert result['max_tokens'] == 128

    def test_fills_missing_params_from_cli(self):
        plugin = _make_plugin()
        body = {'messages': _messages()}
        result = plugin.build_request(body)
        assert result['max_tokens'] == 2048
        assert result['temperature'] == 0.0
        assert result['model'] == 'test-model'

    def test_body_model_preserved(self):
        plugin = _make_plugin()
        body = {'model': 'other-model', 'messages': _messages()}
        result = plugin.build_request(body)
        assert result['model'] == 'other-model'

    def test_does_not_mutate_input_body(self):
        """build_request must not mutate the caller's dict (it copies via dict())."""
        plugin = _make_plugin()
        body = {'messages': _messages()}
        plugin.build_request(body)
        assert body == {'messages': _messages()}


# ---------------------------------------------------------------------------
# Stream / stream_options handling
# ---------------------------------------------------------------------------

class TestStreamOptions:
    """stream_options must only be present when the effective stream is True."""

    def test_body_stream_false_no_stream_options(self):
        plugin = _make_plugin()  # CLI stream=True
        body = {'messages': _messages(), 'stream': False}
        result = plugin.build_request(body)
        assert result['stream'] is False
        assert 'stream_options' not in result

    def test_body_stream_true_gets_stream_options(self):
        plugin = _make_plugin()
        body = {'messages': _messages(), 'stream': True}
        result = plugin.build_request(body)
        assert result['stream'] is True
        assert result['stream_options'] == {'include_usage': True}

    def test_missing_stream_filled_from_cli(self):
        plugin = _make_plugin()  # CLI stream=True
        body = {'messages': _messages()}
        result = plugin.build_request(body)
        assert result['stream'] is True
        assert result['stream_options'] == {'include_usage': True}

    def test_body_stream_true_preserved_over_cli_no_stream(self):
        """fill semantics: body stream=true wins over --no-stream."""
        plugin = _make_plugin(stream=False)
        body = {'messages': _messages(), 'stream': True}
        result = plugin.build_request(body)
        assert result['stream'] is True
        assert result['stream_options'] == {'include_usage': True}

    def test_preserves_existing_stream_options(self):
        plugin = _make_plugin()
        body = {'messages': _messages(), 'stream': True, 'stream_options': {'include_usage': False}}
        result = plugin.build_request(body)
        assert result['stream_options'] == {'include_usage': False}


# ---------------------------------------------------------------------------
# Non-dict inputs still use override semantics (CLI params applied)
# ---------------------------------------------------------------------------

class TestNonDictInputsUseOverride:
    """messages-list and plain-text inputs keep the original override behavior."""

    def test_messages_list_uses_cli_params(self):
        plugin = _make_plugin()
        result = plugin.build_request(_messages())
        assert result['messages'] == _messages()
        assert result['max_tokens'] == 2048
        assert result['temperature'] == 0.0
        assert result['model'] == 'test-model'

    def test_plain_string_prompt(self):
        plugin = _make_plugin()
        result = plugin.build_request('hello')
        assert result['prompt'] == 'hello'
        assert result['model'] == 'test-model'
