"""Tests for the atomic, checksum-verified download helper in download_utils."""

import hashlib
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import patch

from evalscope.utils.download_utils import download_url, file_sha256
from evalscope.utils.url_utils import SUPPORTED_VIDEO_FORMATS
from evalscope.utils.url_utils import download_url as legacy_download_url
from evalscope.utils.url_utils import guess_video_format, video_as_data_uri

PAYLOAD = b'hello evalscope'
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
URL = 'https://example.com/data.csv'


class _FakeResponse:
    """Minimal stand-in for a streaming requests.Response."""

    def __init__(self, payload: bytes, delay: float = 0.0):
        self._payload = payload
        self._delay = delay
        self.headers = {'content-length': str(len(payload))}

    def __enter__(self) -> '_FakeResponse':
        return self

    def __exit__(self, *args) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int = 8192):
        if self._delay:
            time.sleep(self._delay)
        yield self._payload


def test_download_with_sha256_success(tmp_path: Path):
    save_path = tmp_path / 'nested' / 'data.csv'

    with patch('requests.get', return_value=_FakeResponse(PAYLOAD)) as get:
        download_url(URL, str(save_path), sha256=PAYLOAD_SHA256)

    assert get.call_count == 1
    assert save_path.read_bytes() == PAYLOAD
    assert file_sha256(str(save_path)) == PAYLOAD_SHA256
    # Only the final file remains; the temporary file was moved into place.
    assert [entry.name for entry in save_path.parent.iterdir()] == ['data.csv']


def test_download_sha256_mismatch_raises(tmp_path: Path):
    save_path = tmp_path / 'data.csv'

    with patch('requests.get', return_value=_FakeResponse(PAYLOAD)) as get, patch('time.sleep'):
        with pytest.raises(RuntimeError, match='Failed to download'):
            download_url(URL, str(save_path), num_retries=2, sha256='0' * 64)

    assert get.call_count == 2
    assert not save_path.exists()
    # Failed attempts must not leave temporary files behind.
    assert list(tmp_path.iterdir()) == []


def test_existing_valid_file_skips_download(tmp_path: Path):
    save_path = tmp_path / 'data.csv'
    save_path.write_bytes(PAYLOAD)

    with patch('requests.get') as get, patch('requests.head') as head:
        download_url(URL, str(save_path), sha256=PAYLOAD_SHA256)

    get.assert_not_called()
    head.assert_not_called()
    assert save_path.read_bytes() == PAYLOAD


def test_force_download_replaces_same_sized_file(tmp_path: Path):
    save_path = tmp_path / 'data.csv'
    save_path.write_bytes(b'old payload!!!!')
    assert save_path.stat().st_size == len(PAYLOAD)

    with patch('requests.get', return_value=_FakeResponse(PAYLOAD)) as get, patch('requests.head') as head:
        download_url(URL, str(save_path), force=True)

    get.assert_called_once()
    head.assert_not_called()
    assert save_path.read_bytes() == PAYLOAD


def test_concurrent_downloads_fetch_once(tmp_path: Path):
    save_path = tmp_path / 'data.csv'

    with patch('requests.get', return_value=_FakeResponse(PAYLOAD, delay=0.2)) as get:
        threads = [
            threading.Thread(target=download_url, args=(URL, str(save_path)), kwargs={'sha256': PAYLOAD_SHA256})
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert get.call_count == 1
    assert save_path.read_bytes() == PAYLOAD


def test_url_utils_keeps_legacy_download_and_video_exports():
    assert legacy_download_url is download_url
    assert 'mp4' in SUPPORTED_VIDEO_FORMATS
    assert guess_video_format('clip.mov') == 'mov'
    assert video_as_data_uri('data:video/mp4;base64,AA==') == 'data:video/mp4;base64,AA=='
