"""Atomic, checksum-verified file download helpers."""

import hashlib
import os
import tempfile
import threading
import time
from typing import Dict, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()

_download_locks: Dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def file_sha256(path: str) -> str:
    """Compute the SHA-256 hex digest of a file, reading in 1MB chunks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _get_download_lock(save_path: str) -> threading.Lock:
    """Return a per-destination lock so concurrent callers download each file once."""
    with _download_locks_guard:
        lock = _download_locks.get(save_path)
        if lock is None:
            lock = threading.Lock()
            _download_locks[save_path] = lock
        return lock


def _can_reuse_existing_file(url: str, save_path: str, sha256: Optional[str]) -> bool:
    """Decide whether an already-downloaded file at save_path can be reused.

    Verifies the checksum when one is given; otherwise falls back to comparing
    the local file size against a lightweight HEAD request to the URL.
    """
    import requests

    if not os.path.exists(save_path):
        return False
    if sha256 is not None:
        return file_sha256(save_path) == sha256
    try:
        head = requests.head(url, timeout=10, allow_redirects=True)
        remote_size = int(head.headers.get('content-length', 0))
        return remote_size > 0 and os.path.getsize(save_path) == remote_size
    except Exception as e:
        logger.warning(f'HEAD request failed for {url}, will attempt full download: {e}')
        return False


def _download_once(
    url: str, save_path: str, sha256: Optional[str], timeout: int, headers: Optional[Dict[str, str]]
) -> None:
    """Stream the URL to a temporary file, verify it, and atomically move it into place."""
    import requests
    from tqdm import tqdm

    tmp_path: Optional[str] = None
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with (
                tempfile.NamedTemporaryFile(dir=os.path.dirname(save_path), delete=False) as f,
                tqdm(
                    desc=os.path.basename(save_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                tmp_path = f.name
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        bar.update(f.write(chunk))
        if sha256 is not None:
            actual = file_sha256(tmp_path)
            if actual != sha256:
                raise ValueError(f'Checksum mismatch for {url}: expected sha256 {sha256}, got {actual}.')
        os.replace(tmp_path, save_path)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_url(
    url: str,
    save_path: str,
    num_retries: int = 3,
    sha256: Optional[str] = None,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
    force: bool = False,
) -> None:
    """
    Download a file from a URL to a local path with retries.

    The file is streamed to a temporary file in the destination directory and
    atomically moved into place, so readers never observe a partial download.
    Concurrent callers targeting the same save_path are serialized by a
    per-destination thread lock and the file is downloaded only once.

    Args:
        url (str): The URL to download from.
        save_path (str): The local file path to save the downloaded file.
        num_retries (int): Number of times to retry on failure.
        sha256 (Optional[str]): Expected SHA-256 hex digest. When provided, an
            existing valid file skips the download and every downloaded file is
            verified before being moved into place.
        timeout (int): Timeout in seconds for the download request.
        headers (Optional[Dict[str, str]]): Extra HTTP headers for the requests.
        force (bool): Download even when an existing file appears complete.
    """
    save_path = os.path.abspath(save_path)

    # Check if the file already exists before opening any network connection.
    if not force and _can_reuse_existing_file(url, save_path, sha256):
        logger.info(f'File {save_path} already exists and is complete. Skipping download.')
        return

    with _get_download_lock(save_path):
        # Another thread may have completed the download while we waited.
        if not force and _can_reuse_existing_file(url, save_path, sha256):
            logger.info(f'File {save_path} already exists and is complete. Skipping download.')
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for attempt in range(num_retries):
            try:
                logger.info(f'Downloading {url} to {save_path} (attempt {attempt + 1}/{num_retries})...')
                _download_once(url, save_path, sha256, timeout, headers)
                logger.info(f'Downloaded {url} to {save_path}')
                return
            except Exception as e:
                logger.warning(f'Attempt {attempt + 1} failed to download {url}: {e}')
                if attempt < num_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

    raise RuntimeError(f'Failed to download {url} after {num_retries} attempts.')
