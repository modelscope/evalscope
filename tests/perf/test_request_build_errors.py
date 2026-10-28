"""Tests for fatal request-building errors in perf (issue #1565).

``build_request()`` used to turn any exception into ``return None``, so a configuration error
(malformed ``--query-template``, unusable tokenizer) was logged once and the run continued with a
``None`` request.  ``test_connection()`` built its probe request inside the retry loop, so the same
error was retried every 10s until ``--total-timeout`` and then reported as a connection timeout.
"""

import asyncio
import json
import time

import pytest

from evalscope.perf.arguments import Arguments

# Imported as a module: `test_connection` would otherwise be collected as a test case.
from evalscope.perf.core import http_client
from evalscope.perf.plugin.api import openai_api
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin

MESSAGES = [{'role': 'user', 'content': 'hello'}]


class NoTemplateTokenizer:
    """Tokenizer without a usable chat template -- like a base checkpoint (see #1548)."""

    name_or_path = '/models/deepseek-v4-flash'

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        raise ValueError('cannot use chat template functions because tokenizer.chat_template is not set')


def _args(**kwargs) -> Arguments:
    defaults = dict(
        model='test-model',
        url='http://127.0.0.1:59999/v1/chat/completions',
        api='openai',
        dataset='openqa',
        total_timeout=30,
    )
    defaults.update(kwargs)
    args = Arguments(**defaults)
    # run_benchmark() passes a per-sweep int; Arguments normalises the CLI value to a list.
    args.parallel = 1
    return args


class TestBuildRequestPropagates:

    def test_malformed_query_template_raises(self):
        plugin = OpenaiPlugin(_args(query_template='{"stream": true, oops-not-json'))
        with pytest.raises(json.JSONDecodeError):
            plugin.build_request(MESSAGES)

    def test_missing_query_template_file_raises(self, tmp_path):
        missing = tmp_path / 'nope.json'
        plugin = OpenaiPlugin(_args(query_template=f'@{missing}'))
        with pytest.raises(FileNotFoundError):
            plugin.build_request(MESSAGES)

    def test_chat_template_error_reaches_the_caller(self, monkeypatch):
        """The actionable error added in #1564 must survive the --tokenize-prompt path."""
        monkeypatch.setattr(openai_api, 'load_tokenizer', lambda path: NoTemplateTokenizer())
        plugin = OpenaiPlugin(_args(tokenize_prompt=True, tokenizer_path='fake'))
        with pytest.raises(ValueError) as excinfo:
            plugin.build_request(MESSAGES)
        assert 'Failed to apply the chat template' in str(excinfo.value)

    def test_valid_request_is_unaffected(self):
        request = OpenaiPlugin(_args()).build_request(MESSAGES)
        assert request['messages'] == MESSAGES
        assert request['model'] == 'test-model'


class TestConnectionDoesNotRetryBuildErrors:

    def test_build_error_aborts_immediately(self):
        args = _args(query_template='{"stream": true, oops-not-json')
        plugin = OpenaiPlugin(args)
        start = time.perf_counter()
        with pytest.raises(json.JSONDecodeError):
            asyncio.run(http_client.test_connection(args, plugin))
        # The buggy version slept 10s per retry until total_timeout.
        assert time.perf_counter() - start < 5

    def test_none_request_aborts_immediately(self, monkeypatch):
        """Plugins that return None for unusable input must not be retried either."""
        args = _args()
        plugin = OpenaiPlugin(args)
        plugin.build_request = lambda messages, param=None: None
        errors = []
        # evalscope's logger does not propagate to caplog, so capture the call directly.
        monkeypatch.setattr(http_client.logger, 'error', lambda msg, *a, **kw: errors.append(msg))
        start = time.perf_counter()
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        assert time.perf_counter() - start < 5
        # The message names the plugin rather than guessing OpenAI-specific options.
        assert 'OpenaiPlugin' in errors[0]


class _FakeClient500:
    """AioHttpClient stand-in whose every post() returns a retryable 500."""

    def __init__(self, args, api_plugin):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, body):
        from evalscope.perf.utils.benchmark_util import BenchmarkData
        out = BenchmarkData()
        out.success = False
        out.status_code = 500
        out.error = 'Internal Server Error'
        return out


class TestConnectionTestTimeoutBound:

    def test_retryable_5xx_gives_up_within_connection_timeout(self, monkeypatch):
        """A retryable 5xx must give up within --connection-timeout, not --total-timeout.

        Before the fix, ``test_connection`` reused ``--total-timeout`` (default 6h) as the retry
        bound, so a misconfigured URL pointing at a 5xx service waited the full 6h. Now it uses
        ``--connection-timeout``; ``--total-timeout`` is left to govern a single request, as its
        help text already claims.

        We pin connection_timeout to 1s and collapse the 10s backoff to 0, so the loop spins on
        the 5xx and must give up in ~1s of wall-clock. With the old code (bound = total_timeout =
        6h) this would run for hours.
        """
        args = _args(total_timeout=6 * 60 * 60, connection_timeout=1)
        plugin = OpenaiPlugin(args)
        monkeypatch.setattr(http_client, 'AioHttpClient', _FakeClient500)
        # Bind the real asyncio.sleep before patching so the lambda doesn't recurse.
        real_sleep = asyncio.sleep
        monkeypatch.setattr(http_client.asyncio, 'sleep', lambda _: real_sleep(0))
        errors = []
        monkeypatch.setattr(http_client.logger, 'error', lambda msg, *a, **kw: errors.append(msg))
        start = time.perf_counter()
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        elapsed = time.perf_counter() - start
        # Gives up within the ~1s connection bound, not the 6h request timeout.
        assert elapsed < 5
        assert any('timed out after 1s' in e for e in errors)

    def test_default_connection_timeout_is_short(self):
        """The default --connection-timeout must be a short bound, independent of total_timeout."""
        # _args() pins total_timeout=30; the dataclass default for connection_timeout is 60.
        args = Arguments(model='test-model', url='http://127.0.0.1:59999/v1/chat/completions',
                        api='openai', dataset='openqa')
        assert args.connection_timeout == 60
        # The fix's whole point: the connection bound must be its own knob, not 6h.
        assert args.connection_timeout < 6 * 60 * 60
