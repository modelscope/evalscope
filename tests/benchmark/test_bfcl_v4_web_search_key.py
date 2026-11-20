"""BFCL-v4 web-search subsets must be diagnosed when no SerpAPI key is set.

Without ``SERPAPI_API_KEY`` every search call returns an error payload, so the
web-search subsets score 0 after the full run. The adapter documents omitting
them, but nothing told a user who kept the default subset list. These tests pin
the up-front warning.
"""
import logging

import pytest

from evalscope.benchmarks.bfcl.v4.bfcl_v4_adapter import BFCLV4Adapter


def _warn(monkeypatch, categories, caplog):
    caplog.set_level(logging.WARNING)
    adapter = object.__new__(BFCLV4Adapter)  # avoid the bfcl_eval import in __init__
    adapter._warn_if_web_search_unconfigured(categories)
    return [r.getMessage() for r in caplog.records]


def test_warns_when_web_search_selected_without_key(monkeypatch, caplog):
    monkeypatch.delenv('SERPAPI_API_KEY', raising=False)
    messages = _warn(monkeypatch, ['simple_python', 'web_search_base', 'web_search_no_snippet'], caplog)

    assert len(messages) == 1
    assert 'web_search' in messages[0]
    assert 'SERPAPI_API_KEY' in messages[0]


def test_silent_when_key_is_set(monkeypatch, caplog):
    monkeypatch.setenv('SERPAPI_API_KEY', 'test-key')
    assert _warn(monkeypatch, ['web_search_base'], caplog) == []


def test_silent_when_no_web_search_subset_selected(monkeypatch, caplog):
    monkeypatch.delenv('SERPAPI_API_KEY', raising=False)
    assert _warn(monkeypatch, ['simple_python', 'multi_turn_base', 'memory_kv'], caplog) == []


def test_empty_subset_list_is_silent(monkeypatch, caplog):
    monkeypatch.delenv('SERPAPI_API_KEY', raising=False)
    assert _warn(monkeypatch, [], caplog) == []
