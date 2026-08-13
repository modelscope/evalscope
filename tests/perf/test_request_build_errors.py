"""Tests for fatal request-building errors in perf (issue #1565).

``build_request()`` used to turn any exception into ``return None``, so a configuration error
(malformed ``--query-template``, unusable tokenizer) was logged once and the run continued with a
``None`` request.  ``test_connection()`` built its probe request inside the retry loop, so the same
error was retried every 10s until ``--total-timeout`` and then reported as a connection timeout.
"""

import asyncio
import json
import pytest
import time

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
