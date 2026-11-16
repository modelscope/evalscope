"""Fingerprinting and ETag helpers for the predictions endpoint.

File fingerprints invalidate revalidation as evaluation appends or replaces
source caches: any change to the underlying review/prediction files yields a
new ETag, so an unchanged subset revalidates as a 304 while a rerun is always
served fresh. No response bodies are retained in-process.
"""

import hashlib
import os
from typing import Tuple

# (mtime_ns, ctime_ns, size) per source file, in a fixed order.
Fingerprint = Tuple[Tuple[str, int, int, int], ...]


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
