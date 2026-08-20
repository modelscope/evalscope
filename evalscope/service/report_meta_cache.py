"""Process-local memoization of report-list metadata.

The report-list endpoint derives a small metadata dict from each report by
reading its JSON files. Those files are immutable audit records once written
(a rerun rewrites them, changing their mtime; a delete removes the directory),
so a fingerprint over the files' ``(path, mtime, ctime, size)`` and the run
mtime is a lightweight version key for the metadata inputs.

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

# Per-path identity: relative path, mtime (ns), ctime (ns), and size.
Fingerprint = Tuple[Tuple[str, int, int, int], ...]

# cache key -> (fingerprint, metadata). One entry per (root, reference),
# replaced when the fingerprint changes. Per root the cache is bounded by the
# report directories on disk and pruned every request; across roots it grows
# with the number of distinct outputs roots the process serves, which is small
# and stable in practice (the endpoint only lists existing directories).
_CACHE: Dict[str, Tuple[Fingerprint, dict]] = {}
_LOCK = threading.Lock()

# Type of the callable that reads a report and builds its metadata.
MetaBuilder = Callable[[ReportRef, str], Optional[dict]]


def _cache_key_prefix(root: str) -> str:
    """Namespace for one outputs root, so entries never cross between roots."""
    return f'{os.path.realpath(root)}\x00'


def _cache_key(root: str, ref: ReportRef) -> str:
    return f'{_cache_key_prefix(root)}{ref.key}'


def report_ref_fingerprint(root: str, ref: ReportRef) -> Fingerprint:
    """Version key for the report JSONs and timestamp fallback inputs."""
    model_dir = report_model_dir(root, ref)
    entries: List[Tuple[str, int, int, int]] = []
    run_dir = os.path.join(root, ref.run_id)
    try:
        run_stat = os.stat(run_dir)
        entries.append(('@run', run_stat.st_mtime_ns, run_stat.st_ctime_ns, run_stat.st_size))
    except OSError:
        pass

    for file_path in glob.glob(os.path.join(model_dir, '**', '*.json'), recursive=True):
        if os.path.basename(file_path) == DataCollection.REPORT_NAME:
            continue
        try:
            st = os.stat(file_path)
        except OSError:
            continue
        entries.append((os.path.relpath(file_path, model_dir), st.st_mtime_ns, st.st_ctime_ns, st.st_size))
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
        observed = _CACHE.get(key)
        if observed is not None and observed[0] == fingerprint:
            return observed[1]

    meta = compute(ref, root)
    if meta is None:
        return None

    with _LOCK:
        current = _CACHE.get(key)
        if current is observed or current is None or current[0] == fingerprint:
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
    filter/sort/page values that also shape the response. The digest is built
    incrementally so no O(N) intermediate payload is allocated.
    """
    digest = hashlib.sha256()
    for key, fingerprint in sorted(fingerprints):
        digest.update(f'{key}:{fingerprint}\n'.encode('utf-8'))
    for part in query_parts:
        digest.update(f'q:{part}\n'.encode('utf-8'))
    return digest.hexdigest()


def clear_report_meta_cache() -> None:
    """Drop all cached entries. Intended for tests."""
    with _LOCK:
        _CACHE.clear()
