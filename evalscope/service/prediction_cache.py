"""Process-local cache for the predictions endpoint.

Loading predictions for one model/dataset/subset reads the whole review JSONL
(and the prediction JSONL for the perf fallback), deserialises every sample and
builds a DataFrame. The response can be hundreds of MB, and every browse of the
same report repeats all of it. Those cache files are immutable audit records
once written (a rerun rewrites them, changing their mtime; a delete removes the
directory), so a fingerprint over the source files' ``(mtime, ctime, size)`` is
a lightweight version key.

This mirrors ``report_meta_cache`` and is used two ways in the predictions
endpoint:

* as an ETag so an unchanged response revalidates as ``304`` (no re-transfer of
  a large body), and
* as the key for a small process-local memo of the computed records (no
  re-read / re-deserialise while the files are unchanged).
"""

import hashlib
import os
import threading
from typing import Any, List, Optional, Tuple

# (mtime_ns, ctime_ns, size) per source file, in a fixed order.
Fingerprint = Tuple[Tuple[str, int, int, int], ...]

# cache key -> (fingerprint, records). One entry per
# (root, run_id, model_id, dataset, subset); replaced when the fingerprint
# changes. Bounded in practice by the report subsets a process serves.
_CACHE: dict[str, Tuple[Fingerprint, List[Any]]] = {}
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


def get_cached_records(cache_key: str, fingerprint: Fingerprint) -> Optional[List[Any]]:
    """Return memoised records when the fingerprint matches, else ``None``."""
    with _LOCK:
        observed = _CACHE.get(cache_key)
        if observed is not None and observed[0] == fingerprint:
            return observed[1]
    return None


def store_records(cache_key: str, fingerprint: Fingerprint, records: List[Any]) -> None:
    """Memoise the computed records under the current fingerprint."""
    with _LOCK:
        _CACHE[cache_key] = (fingerprint, records)


def clear_prediction_cache() -> None:
    """Drop all cached entries. Intended for tests."""
    with _LOCK:
        _CACHE.clear()
