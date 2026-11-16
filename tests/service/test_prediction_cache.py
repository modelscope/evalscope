"""The predictions memo retains at most its payload budget and releases stale entries."""

import os
from pathlib import Path
from typing import Iterator

import pytest

from evalscope.service import prediction_cache as cache

FINGERPRINT = (('@review', 1, 1, 10), ('@prediction', 1, 1, 10))


@pytest.fixture(autouse=True)
def isolated_cache() -> Iterator[None]:
    cache.clear_prediction_cache()
    yield
    cache.clear_prediction_cache()


def test_byte_budget_evicts_the_least_recently_used_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, '_MAX_CACHE_BYTES', 8)
    for key in ('a', 'b'):
        cache.store_response(key, FINGERPRINT, b'1234')
    assert cache.get_cached_response('a', FINGERPRINT) == b'1234'

    cache.store_response('c', FINGERPRINT, b'5678')

    assert cache.get_cached_response('b', FINGERPRINT) is None
    assert cache.get_cached_response('a', FINGERPRINT) == b'1234'
    assert cache.get_cached_response('c', FINGERPRINT) == b'5678'


def test_entry_limit_bounds_small_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, '_MAX_CACHE_ENTRIES', 2)
    for key in ('a', 'b', 'c'):
        cache.store_response(key, FINGERPRINT, b'')
    assert cache.get_cached_response('a', FINGERPRINT) is None
    assert cache.get_cached_response('b', FINGERPRINT) == b''
    assert cache.get_cached_response('c', FINGERPRINT) == b''


def test_oversized_replacement_releases_old_body_without_evicting_others(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, '_MAX_CACHE_BYTES', 8)
    cache.store_response('a', FINGERPRINT, b'1234')
    cache.store_response('b', FINGERPRINT, b'5678')

    cache.store_response('a', FINGERPRINT, b'oversized')

    assert cache.get_cached_response('a', FINGERPRINT) is None
    assert cache.get_cached_response('b', FINGERPRINT) == b'5678'
    cache.store_response('c', FINGERPRINT, b'new!')
    assert cache.get_cached_response('b', FINGERPRINT) == b'5678'
    assert cache.get_cached_response('c', FINGERPRINT) == b'new!'


def test_new_fingerprint_releases_old_response() -> None:
    cache.store_response('a', FINGERPRINT, b'old')
    updated = (('@review', 2, 2, 11), ('@prediction', 1, 1, 10))
    assert cache.get_cached_response('a', updated) is None
    assert cache.get_cached_response('a', FINGERPRINT) is None
    cache.store_response('a', updated, b'new')
    assert cache.get_cached_response('a', updated) == b'new'


def test_report_cleanup_preserves_other_models_and_roots(tmp_path: Path) -> None:
    root = os.path.realpath(tmp_path)
    keys = [f'{root}\x00run\x00model\x00ds\x00{subset}' for subset in ('a', 'b')]
    others = [f'{root}\x00run\x00other\x00ds\x00a', f'{root}-other\x00run\x00model\x00ds\x00a']
    for key in keys + others:
        cache.store_response(key, FINGERPRINT, b'body')

    cache.discard_prediction_cache(root, 'run', 'model')

    for key in keys:
        assert cache.get_cached_response(key, FINGERPRINT) is None
    for key in others:
        assert cache.get_cached_response(key, FINGERPRINT) == b'body'
