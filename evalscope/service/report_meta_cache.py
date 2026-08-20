"""Process-local memoization of report-list metadata.

The report-list endpoint derives a small metadata dict from each report by
reading its JSON files. Those files are immutable audit records once written
(a rerun rewrites them, changing their mtime; a delete removes the directory),
so a fingerprint over the files' ``(path, mtime, size)`` is a sound version key:
an unchanged fingerprint guarantees unchanged derived metadata.

This lets every list request stat the report files (cheap) while reading and
parsing them (expensive) only for references that are new or have changed since
the last request. Entries are keyed by ``(root, ref)`` so switching the outputs
root never makes one root's entries evict or shadow another's.
"""
import glob
import hashlib
import os
import threading
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from evalscope.constants import DataCollection
from evalscope.report import ReportRef
from evalscope.utils.data_utils import report_model_dir

# Per-file identity: relative path, mtime (ns) and size. The tuple of these over
# a reference's JSON files is its version key.
Fingerprint = Tuple[Tuple[str, int, int], ...]

# cache key -> (fingerprint, meta_or_None). One entry per (root, reference),
# replaced when the fingerprint changes, so the cache is bounded by the number
# of report directories on disk. ``None`` metadata (an unreadable report) is
# cached too, so a broken report is not re-read on every request.
_CACHE: Dict[str, Tuple[Fingerprint, Optional[dict]]] = {}
_LOCK = threading.Lock()

# Type of the callable that reads a report and builds its metadata.
MetaBuilder = Callable[[ReportRef, str], Optional[dict]]


def _cache_key_prefix(root: str) -> str:
    """Namespace for one outputs root, so entries never cross between roots."""
    return f'{os.path.realpath(root)}\x00'


def _cache_key(root: str, ref: ReportRef) -> str:
    return f'{_cache_key_prefix(root)}{ref.key}'


def report_ref_fingerprint(root: str, ref: ReportRef) -> Fingerprint:
    """Version key for one report reference: its files' paths, mtimes and sizes.

    Stat-only, no file is read. The collection report is skipped to match
    ``get_report_list``'s own exclusion, so the fingerprint tracks exactly the
    files the metadata is derived from.
    """
    model_dir = report_model_dir(root, ref)
    entries: List[Tuple[str, int, int]] = []
    for file_path in glob.glob(os.path.join(model_dir, '**', '*.json'), recursive=True):
        if os.path.basename(file_path) == DataCollection.REPORT_NAME:
            continue
        try:
            st = os.stat(file_path)
        except OSError:
            continue
        entries.append((os.path.relpath(file_path, model_dir), st.st_mtime_ns, st.st_size))
    entries.sort()
    return tuple(entries)


def build_report_meta_cached(root: str, ref: ReportRef, fingerprint: Fingerprint,
                             compute: MetaBuilder) -> Optional[dict]:
    """Return the cached metadata for ``ref`` or compute and store it.

    ``fingerprint`` is supplied by the caller (which already needs it for the
    list ETag) so the file stat is not repeated here. A hit reuses the stored
    metadata; a miss calls ``compute`` and writes the result back.
    """
    key = _cache_key(root, ref)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

    # Computed outside the lock to keep file I/O off it; a rare double-compute on
    # concurrent misses is harmless because the result is idempotent.
    meta = compute(ref, root)

    with _LOCK:
        _CACHE[key] = (fingerprint, meta)
    return meta


def prune_report_meta_cache(root: str, valid_ref_keys: Iterable[str]) -> None:
    """Drop entries for ``root`` whose reference is no longer present on disk.

    Other roots' entries are left untouched, so alternating between roots does
    not evict either one's warm cache.
    """
    prefix = _cache_key_prefix(root)
    keep = {f'{prefix}{ref_key}' for ref_key in valid_ref_keys}
    with _LOCK:
        for key in [k for k in _CACHE if k.startswith(prefix) and k not in keep]:
            del _CACHE[key]


def list_etag(fingerprints: Sequence[Tuple[str, Fingerprint]], query_parts: Sequence[str]) -> str:
    """Digest identifying a list response: the scanned set plus the query.

    ``fingerprints`` is the ``(ref.key, fingerprint)`` pairs gathered while
    serving the list, so this adds no file reads. ``query_parts`` folds in the
    filter/sort/page values that also shape the response.
    """
    parts: List[str] = sorted(f'{key}:{fp}' for key, fp in fingerprints)
    parts.extend(f'q:{part}' for part in query_parts)
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()


def clear_report_meta_cache() -> None:
    """Drop all cached entries. Intended for tests."""
    with _LOCK:
        _CACHE.clear()
