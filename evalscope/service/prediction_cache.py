"""Bounded process-local memo of validated JSON predictions responses.

File fingerprints invalidate responses as evaluation appends or replaces source
caches. Only serialized bytes are retained, so the byte budget measures the cached
payload and hits avoid rebuilding both the DataFrame and the JSON representation.
"""

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Optional, Tuple

# (mtime_ns, ctime_ns, size) per source file, in a fixed order.
Fingerprint = Tuple[Tuple[str, int, int, int], ...]

_MAX_CACHE_BYTES = 64 * 1024 * 1024
_MAX_CACHE_ENTRIES = 16
_CACHE: OrderedDict[str, Tuple[Fingerprint, bytes]] = OrderedDict()
_CACHE_BYTES = 0
_LOCK = threading.Lock()


def _stat_entry(name: str, path: str) -> Tuple[str, int, int, int]:
    """Fingerprint one source file; a missing file records zeros."""
    try:
        st = os.stat(path)
    except OSError:
        return (name, 0, 0, 0)
    return (name, st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def prediction_fingerprint(review_path: str, prediction_path: str) -> Fingerprint:
    """Version key over the two files the predictions response is derived from."""
    return (_stat_entry('@review', review_path), _stat_entry('@prediction', prediction_path))


def prediction_etag(cache_key: str, fingerprint: Fingerprint) -> str:
    """Stable ETag for a predictions response: its identity plus its fingerprint."""
    digest = hashlib.sha256()
    digest.update(cache_key.encode('utf-8'))
    for name, mtime_ns, ctime_ns, size in fingerprint:
        digest.update(f'\x00{name}:{mtime_ns}:{ctime_ns}:{size}'.encode('utf-8'))
    return digest.hexdigest()


def get_cached_response(cache_key: str, fingerprint: Fingerprint) -> Optional[bytes]:
    """Return validated response bytes on a hit and release outdated entries."""
    global _CACHE_BYTES
    with _LOCK:
        observed = _CACHE.get(cache_key)
        if observed is not None and observed[0] == fingerprint:
            _CACHE.move_to_end(cache_key)
            return observed[1]
        if observed is not None:
            _CACHE_BYTES -= len(_CACHE.pop(cache_key)[1])
    return None


def store_response(cache_key: str, fingerprint: Fingerprint, body: bytes) -> None:
    """Memoise a validated response, evicting least recently used payloads."""
    global _CACHE_BYTES
    with _LOCK:
        previous = _CACHE.pop(cache_key, None)
        if previous is not None:
            _CACHE_BYTES -= len(previous[1])
        if len(body) > _MAX_CACHE_BYTES or _MAX_CACHE_ENTRIES <= 0:
            return
        while _CACHE and (_CACHE_BYTES + len(body) > _MAX_CACHE_BYTES or len(_CACHE) >= _MAX_CACHE_ENTRIES):
            _, (_, evicted) = _CACHE.popitem(last=False)
            _CACHE_BYTES -= len(evicted)
        _CACHE[cache_key] = (fingerprint, body)
        _CACHE_BYTES += len(body)


def discard_prediction_cache(root: str, run_id: str, model_id: str) -> None:
    """Release responses belonging to one deleted report, leaving other roots intact."""
    global _CACHE_BYTES
    prefix = f'{os.path.realpath(root)}\x00{run_id}\x00{model_id}\x00'
    with _LOCK:
        for key in [key for key in _CACHE if key.startswith(prefix)]:
            _CACHE_BYTES -= len(_CACHE.pop(key)[1])


def clear_prediction_cache() -> None:
    """Drop all cached entries. Intended for tests."""
    global _CACHE_BYTES
    with _LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0
