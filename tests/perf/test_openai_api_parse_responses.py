"""Unit tests for ``OpenaiPlugin.parse_responses``.

These cover the behavior of the usage-block parsing branch, including
the cases where an OpenAI-compatible endpoint returns a partial usage
block (e.g. Vertex AI's Gemini 2.5 reasoning mode omits
``completion_tokens`` from ``usage`` when ``max_tokens`` is reached
with only reasoning tokens emitted).
"""
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin


@pytest.fixture
def plugin():
    """Return an ``OpenaiPlugin`` instance with no tokenizer configured.

    Using ``tokenizer_path=None`` lets us assert that the usage-block
    branch returns directly when ``prompt_tokens`` or
    ``completion_tokens`` is present, without falling through to the
    content-based tokenization path (which would raise ``ValueError``
    because no tokenizer is available).
    """
    args = Arguments(model='test-model', api='openai', number=1, parallel=1)
    return OpenaiPlugin(args)


def test_parse_responses_full_usage(plugin):
    """Standard case: both prompt_tokens and completion_tokens present."""
    responses = [{'usage': {'prompt_tokens': 100, 'completion_tokens': 50}}]
    assert plugin.parse_responses(responses) == (100, 50)


def test_parse_responses_missing_completion_tokens(plugin):
    """Vertex Gemini 2.5 reasoning-mode shape: completion_tokens omitted.

    Before this fix, ``KeyError`` on the missing key was caught by the
    broad ``except`` and the parser returned ``(0, 0)``, silently
    discarding the valid ``prompt_tokens``.
    """
    responses = [{
        'choices': [{
            'finish_reason': 'length',
            'index': 0
        }],
        'usage': {
            'prompt_tokens': 7180,
            'total_tokens': 7193,
            'completion_tokens_details': {
                'reasoning_tokens': 13
            },
        },
    }]
    assert plugin.parse_responses(responses) == (7180, 0)


def test_parse_responses_missing_prompt_tokens(plugin):
    """Symmetric case: only ``completion_tokens`` is present."""
    responses = [{'usage': {'completion_tokens': 25}}]
    assert plugin.parse_responses(responses) == (0, 25)


def test_parse_responses_null_completion_tokens(plugin):
    """``completion_tokens: null`` is treated the same as missing."""
    responses = [{'usage': {'prompt_tokens': 42, 'completion_tokens': None}}]
    assert plugin.parse_responses(responses) == (42, 0)


def test_parse_responses_explicit_zero_tokens(plugin):
    """Explicit ``0`` values are returned, not treated as missing.

    Falling through to content-based tokenization here would raise
    ``ValueError`` when no tokenizer is configured. The
    ``'prompt_tokens' in usage or 'completion_tokens' in usage`` check
    guarantees we honor explicit zeros and avoid the fall-through.
    """
    responses = [{'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}]
    assert plugin.parse_responses(responses) == (0, 0)


def test_parse_responses_empty_response_list(plugin):
    """An empty response list returns ``(0, 0)`` (unchanged behavior)."""
    assert plugin.parse_responses([]) == (0, 0)


def test_parse_responses_stream_last_chunk_usage(plugin):
    """Streaming case: only the final chunk carries the usage block."""
    responses = [
        {
            'choices': [{
                'delta': {
                    'content': 'hello'
                }
            }]
        },
        {
            'choices': [{
                'delta': {
                    'content': ' world'
                }
            }]
        },
        {
            'choices': [{
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 8,
                'completion_tokens': 2
            },
        },
    ]
    assert plugin.parse_responses(responses) == (8, 2)


def test_parse_responses_usage_present_but_empty_falls_through(plugin):
    """An empty usage block (``{}``) should fall through to content parsing.

    With no tokenizer configured and no parseable content, the
    content-based path raises ``ValueError`` — verifying that the
    usage-branch did NOT swallow this case and return ``(0, 0)`` (which
    would mask a missing-tokenizer misconfiguration).
    """
    responses = [{'usage': {}}]
    with pytest.raises(ValueError):
        plugin.parse_responses(responses, request='{}')
