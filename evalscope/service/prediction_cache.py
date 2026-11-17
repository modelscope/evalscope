"""File-based ETag for predictions, without retaining response bodies."""

import hashlib
import os


def prediction_etag(cache_key: str, review_path: str, prediction_path: str) -> str:
    """Hash the response identity and source-file timestamps and sizes."""
    digest = hashlib.sha256()
    digest.update(cache_key.encode('utf-8'))
    for name, path in (('@review', review_path), ('@prediction', prediction_path)):
        try:
            stat = os.stat(path)
            mtime_ns, ctime_ns, size = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
        except OSError:
            mtime_ns, ctime_ns, size = 0, 0, 0
        digest.update(f'\x00{name}:{mtime_ns}:{ctime_ns}:{size}'.encode('utf-8'))
    return digest.hexdigest()
