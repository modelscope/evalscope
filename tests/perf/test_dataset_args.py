"""Tests for --dataset-args body_compose_mode in OpenaiPlugin.build_request."""

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.types import AnnotatedBody


def _make_plugin(dataset_args=None):
    args = Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        dataset_args=dataset_args,
    )
    return OpenaiPlugin(args)


# ---------------------------------------------------------------------------
# AnnotatedBody type tests
# ---------------------------------------------------------------------------

class TestAnnotatedBody:
    """AnnotatedBody dict subclass behavior."""

    def test_default_compose_mode_is_override(self):
        ab = AnnotatedBody({'a': 1})
        assert ab.compose_mode == 'override'

    def test_rejects_invalid_compose_mode(self):
        with pytest.raises(ValueError, match='Invalid compose_mode'):
            AnnotatedBody({}, compose_mode='bogus')


# ---------------------------------------------------------------------------
# Override mode (default)
# ---------------------------------------------------------------------------

class TestBodyComposeModeOverride:
    """Default mode: CLI params overwrite body fields."""

    def test_overwrites_existing_fields(self):
        plugin = _make_plugin()
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 512, 'temperature': 0.7}
        result = plugin.build_request(body)
        assert result['max_tokens'] == 2048
        assert result['temperature'] == 0.0
        assert result['model'] == 'test-model'

    def test_explicit_override_arg(self):
        plugin = _make_plugin(dataset_args={'body_compose_mode': 'override'})
        body = AnnotatedBody(
            {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 512},
            compose_mode='override',
        )
        result = plugin.build_request(body)
        assert result['max_tokens'] == 2048

    def test_plain_dict_always_overrides(self):
        """Plain dict (no AnnotatedBody wrapper) always uses override — backward compat."""
        plugin = _make_plugin()
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 512}
        result = plugin.build_request(body)
        assert result['max_tokens'] == 2048

    def test_no_stream_overrides_body_stream_true(self):
        """--no-stream should override body's stream=True and suppress stream_options."""
        args = Arguments(
            model='test-model',
            url='http://localhost:8080/v1/chat/completions',
            stream=False,
        )
        plugin = OpenaiPlugin(args)
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True}
        result = plugin.build_request(body)
        assert result['stream'] is False
        assert 'stream_options' not in result

    def test_override_overrides_stream_false(self):
        """CLI default stream=True should override body's stream=False."""
        plugin = _make_plugin()
        body = AnnotatedBody(
            {'messages': [{'role': 'user', 'content': 'hi'}], 'stream': False},
            compose_mode='override',
        )
        result = plugin.build_request(body)
        assert result['stream'] is True
        assert 'stream_options' in result

    def test_messages_list_unaffected(self):
        plugin = _make_plugin(dataset_args={'body_compose_mode': 'override'})
        messages = [{'role': 'user', 'content': 'hi'}]
        result = plugin.build_request(messages)
        assert result['max_tokens'] == 2048
        assert result['model'] == 'test-model'


# ---------------------------------------------------------------------------
# Fill mode
# ---------------------------------------------------------------------------

class TestBodyComposeModeFill:
    """Fill mode: body fields preserved, CLI fills missing."""

    def test_preserves_existing_fields(self):
        plugin = _make_plugin()
        body = AnnotatedBody(
            {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 512, 'temperature': 0.7},
            compose_mode='fill',
        )
        result = plugin.build_request(body)
        assert result['max_tokens'] == 512
        assert result['temperature'] == 0.7

    def test_fills_missing_fields(self):
        plugin = _make_plugin()
        body = AnnotatedBody(
            {'messages': [{'role': 'user', 'content': 'hi'}]},
            compose_mode='fill',
        )
        result = plugin.build_request(body)
        assert result['max_tokens'] == 2048
        assert result['model'] == 'test-model'
        assert result['temperature'] == 0.0

    def test_preserves_stream_false(self):
        plugin = _make_plugin()
        body = AnnotatedBody(
            {'messages': [{'role': 'user', 'content': 'hi'}], 'stream': False},
            compose_mode='fill',
        )
        result = plugin.build_request(body)
        assert result['stream'] is False

    def test_no_stream_options_when_stream_false(self):
        """stream=false in body should not get stream_options injected."""
        plugin = _make_plugin()
        body = AnnotatedBody(
            {'messages': [{'role': 'user', 'content': 'hi'}], 'stream': False},
            compose_mode='fill',
        )
        result = plugin.build_request(body)
        assert 'stream_options' not in result

    def test_preserves_stream_options(self):
        plugin = _make_plugin()
        body = AnnotatedBody(
            {
                'messages': [{'role': 'user', 'content': 'hi'}],
                'stream': True,
                'stream_options': {'include_usage': False},
            },
            compose_mode='fill',
        )
        result = plugin.build_request(body)
        assert result['stream_options'] == {'include_usage': False}

    def test_messages_list_unaffected(self):
        plugin = _make_plugin()
        messages = [{'role': 'user', 'content': 'hi'}]
        result = plugin.build_request(messages)
        assert result['max_tokens'] == 2048
        assert result['model'] == 'test-model'


# ---------------------------------------------------------------------------
# Passthrough mode
# ---------------------------------------------------------------------------

class TestBodyComposeModePassthrough:
    """Passthrough mode: body sent as-is, compose skipped."""

    def test_returns_body_unchanged(self):
        plugin = _make_plugin()
        body = AnnotatedBody(
            {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 512},
            compose_mode='passthrough',
        )
        result = plugin.build_request(body)
        assert result == {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 512}
        assert 'model' not in result
        assert 'temperature' not in result
        assert 'stream' not in result

    def test_all_fields_preserved(self):
        plugin = _make_plugin()
        body = AnnotatedBody(
            {
                'model': 'other-model',
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 100,
                'temperature': 0.5,
                'stream': False,
            },
            compose_mode='passthrough',
        )
        result = plugin.build_request(body)
        assert result['model'] == 'other-model'
        assert result['max_tokens'] == 100
        assert result['temperature'] == 0.5
        assert result['stream'] is False

    def test_returns_plain_dict(self):
        """Passthrough result should be a plain dict, not AnnotatedBody."""
        plugin = _make_plugin()
        body = AnnotatedBody({'messages': [{'role': 'user', 'content': 'hi'}]}, compose_mode='passthrough')
        result = plugin.build_request(body)
        assert type(result) is dict

    def test_messages_list_unaffected(self):
        """Messages list still gets compose even if plugin has passthrough dataset_args."""
        plugin = _make_plugin()
        messages = [{'role': 'user', 'content': 'hi'}]
        result = plugin.build_request(messages)
        assert result['max_tokens'] == 2048
        assert result['model'] == 'test-model'


# ---------------------------------------------------------------------------
# Dataset args parsing & validation
# ---------------------------------------------------------------------------

class TestDatasetArgsParsing:
    """Test --dataset-args field on Arguments."""

    def test_none_by_default(self):
        args = Arguments(model='test-model')
        assert args.dataset_args is None

    def test_accepts_dict(self):
        args = Arguments(model='test-model', dataset_args={'body_compose_mode': 'fill'})
        assert args.dataset_args == {'body_compose_mode': 'fill'}


class TestLineByLineArgsValidation:
    """Test LineByLineArgs pydantic model."""

    def test_rejects_unknown_keys(self):
        from evalscope.perf.plugin.datasets.line_by_line import LineByLineArgs
        with pytest.raises(Exception):
            LineByLineArgs(body_compose_mode='fill', bogus=1)

    def test_rejects_invalid_mode(self):
        from evalscope.perf.plugin.datasets.line_by_line import LineByLineArgs
        with pytest.raises(Exception):
            LineByLineArgs(body_compose_mode='invalid')

    def test_default_mode(self):
        from evalscope.perf.plugin.datasets.line_by_line import LineByLineArgs
        config = LineByLineArgs()
        assert config.body_compose_mode == 'override'


# ---------------------------------------------------------------------------
# Messages list with non-override mode
# ---------------------------------------------------------------------------

class TestMessagesListWithNonOverrideMode:
    """Messages list input must not be wrapped in AnnotatedBody."""

    def test_fill_mode_with_messages_list(self):
        """fill mode should pass messages list through unchanged."""
        plugin = _make_plugin(dataset_args={'body_compose_mode': 'fill'})
        messages = [{'role': 'user', 'content': 'hi'}]
        result = plugin.build_request(messages)
        assert result['messages'] == [{'role': 'user', 'content': 'hi'}]
        assert result['model'] == 'test-model'

    def test_passthrough_mode_with_messages_list(self):
        """passthrough mode should pass messages list through unchanged."""
        plugin = _make_plugin(dataset_args={'body_compose_mode': 'passthrough'})
        messages = [{'role': 'user', 'content': 'hi'}]
        result = plugin.build_request(messages)
        assert result['messages'] == [{'role': 'user', 'content': 'hi'}]
        assert result['model'] == 'test-model'
