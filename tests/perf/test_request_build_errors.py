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

    calls = 0  # class-level so the instance created inside test_connection is observable

    def __init__(self, args, api_plugin):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, body, **kwargs):
        from evalscope.perf.utils.benchmark_util import BenchmarkData
        type(self).calls += 1
        out = BenchmarkData()
        out.success = False
        out.status_code = 500
        out.error = 'Internal Server Error'
        return out


class _FakeClientHttpFailure:
    """AioHttpClient stand-in returning a configurable HTTP failure."""

    calls = 0
    status_code = 404
    error = 'Not Found'

    def __init__(self, args, api_plugin):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, body, **kwargs):
        from evalscope.perf.utils.benchmark_util import BenchmarkData
        type(self).calls += 1
        out = BenchmarkData()
        out.success = False
        out.status_code = type(self).status_code
        out.error = type(self).error
        return out


class _FakeClient4xx(_FakeClientHttpFailure):
    pass


class _FakeClient302(_FakeClientHttpFailure):
    status_code = 302
    error = 'Found'


class _FakeClientProgramError(_FakeClientHttpFailure):
    """Match post()'s default failure conversion and preflight propagation option."""

    async def post(self, body, *, propagate_unexpected_errors: bool = False, **kwargs):
        from evalscope.perf.utils.benchmark_util import BenchmarkData
        type(self).calls += 1
        if propagate_unexpected_errors:
            raise ValueError('plugin exploded')
        out = BenchmarkData()
        out.success = False
        out.error = 'plugin exploded'
        return out


class _FakeClientHang:
    """AioHttpClient stand-in whose post() never returns (simulates a hung server)."""

    calls = 0

    def __init__(self, args, api_plugin):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, body, **kwargs):
        type(self).calls += 1
        # Sleep far longer than any test deadline so asyncio.wait_for must cancel it.
        await asyncio.sleep(1000)
        from evalscope.perf.utils.benchmark_util import BenchmarkData
        return BenchmarkData()


class TestConnectionPreflightPolicy:
    """The fixed internal preflight policy: 60s deadline, 3 attempts, 2s interval, with each
    attempt cancelled at the remaining deadline via asyncio.wait_for."""

    def test_repeated_500_stops_after_three_attempts(self, monkeypatch):
        """Immediate retryable 500s must stop after _PREFLIGHT_MAX_ATTEMPTS, not loop to timeout."""
        args = _args()
        plugin = OpenaiPlugin(args)
        _FakeClient500.calls = 0
        monkeypatch.setattr(http_client, 'AioHttpClient', _FakeClient500)
        # Collapse the retry interval so the test doesn't wait.
        real_sleep = asyncio.sleep
        monkeypatch.setattr(http_client.asyncio, 'sleep', lambda _: real_sleep(0))
        errors = []
        monkeypatch.setattr(http_client.logger, 'error', lambda msg, *a, **kw: errors.append(msg))
        start = time.perf_counter()
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        elapsed = time.perf_counter() - start
        # Exactly 3 attempts, then failure — well under the 60s deadline.
        assert _FakeClient500.calls == 3
        assert elapsed < 5
        assert any('exhausted 3 attempts' in e for e in errors)

    def test_hung_request_cancelled_at_deadline(self, monkeypatch):
        """A request that hangs past the deadline is cancelled by asyncio.wait_for, not awaited
        up to --total-timeout."""
        args = _args(total_timeout=6 * 60 * 60)
        plugin = OpenaiPlugin(args)
        _FakeClientHang.calls = 0
        monkeypatch.setattr(http_client, 'AioHttpClient', _FakeClientHang)
        # Shorten the deadline to 0.5s so the test is fast.
        monkeypatch.setattr(http_client, '_PREFLIGHT_DEADLINE', 0.5)
        errors = []
        monkeypatch.setattr(http_client.logger, 'error', lambda msg, *a, **kw: errors.append(msg))
        start = time.perf_counter()
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        elapsed = time.perf_counter() - start
        # The hung post() never returned; wait_for cancelled it at ~0.5s, not 6h.
        assert _FakeClientHang.calls == 1
        assert 0.4 < elapsed < 3
        assert any('hung past the' in e for e in errors)

    def test_4xx_receives_exactly_one_attempt(self, monkeypatch):
        """A non-retryable 4xx must abort after a single attempt, no retries."""
        args = _args()
        plugin = OpenaiPlugin(args)
        _FakeClient4xx.calls = 0
        monkeypatch.setattr(http_client, 'AioHttpClient', _FakeClient4xx)
        errors = []
        monkeypatch.setattr(http_client.logger, 'error', lambda msg, *a, **kw: errors.append(msg))
        start = time.perf_counter()
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        elapsed = time.perf_counter() - start
        # Exactly one attempt, no retry sleep.
        assert _FakeClient4xx.calls == 1
        assert elapsed < 2
        assert any('non-retryable HTTP 404' in e for e in errors)

    def test_3xx_receives_exactly_one_attempt(self, monkeypatch):
        args = _args()
        plugin = OpenaiPlugin(args)
        _FakeClient302.calls = 0
        monkeypatch.setattr(http_client, 'AioHttpClient', _FakeClient302)
        errors = []
        monkeypatch.setattr(http_client.logger, 'error', lambda msg, *a, **kw: errors.append(msg))
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        assert _FakeClient302.calls == 1
        assert any('non-retryable HTTP 302' in e for e in errors)

    def test_unexpected_exception_receives_exactly_one_attempt(self, monkeypatch):
        args = _args()
        plugin = OpenaiPlugin(args)
        _FakeClientProgramError.calls = 0
        monkeypatch.setattr(http_client, 'AioHttpClient', _FakeClientProgramError)
        errors = []
        warnings = []
        monkeypatch.setattr(http_client.logger, 'error', lambda msg, *a, **kw: errors.append(msg))
        monkeypatch.setattr(http_client.logger, 'warning', lambda msg, *a, **kw: warnings.append(msg))
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        assert _FakeClientProgramError.calls == 1
        assert not warnings
        assert any('non-retryable ValueError: plugin exploded' in e for e in errors)

    def test_final_sleep_does_not_exceed_deadline(self, monkeypatch):
        """The retry sleep is clamped to the remaining time, so the last sleep can't push the
        probe past the deadline."""
        args = _args()
        plugin = OpenaiPlugin(args)
        _FakeClient500.calls = 0
        monkeypatch.setattr(http_client, 'AioHttpClient', _FakeClient500)
        # Tight deadline so the inter-attempt sleep would overshoot if not clamped.
        monkeypatch.setattr(http_client, '_PREFLIGHT_DEADLINE', 1)
        monkeypatch.setattr(http_client, '_PREFLIGHT_RETRY_INTERVAL', 2)
        # Track actual sleep durations.
        slept = []
        real_sleep = asyncio.sleep

        async def spy_sleep(seconds):
            slept.append(seconds)
            await real_sleep(0)

        monkeypatch.setattr(http_client.asyncio, 'sleep', spy_sleep)
        start = time.perf_counter()
        assert asyncio.run(http_client.test_connection(args, plugin)) is False
        elapsed = time.perf_counter() - start
        # Remaining time never exceeds the one-second deadline; an unclamped two-second sleep fails.
        assert slept and all(0 < s <= 1 for s in slept)
        assert elapsed < 3
