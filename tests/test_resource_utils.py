import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from evalscope.utils import resource_utils


class _FakeNltkData:

    def __init__(self, available: bool = False):
        self.available = available
        self.lookups = []

    def find(self, resource_path: str) -> str:
        self.lookups.append(resource_path)
        if not self.available:
            raise LookupError(resource_path)
        return resource_path


@pytest.fixture(autouse=True)
def clear_check_nltk_data_cache() -> Iterator[None]:
    resource_utils.check_nltk_data.cache_clear()
    yield
    resource_utils.check_nltk_data.cache_clear()


def _install_fake_nltk(monkeypatch: pytest.MonkeyPatch, data: _FakeNltkData) -> None:
    monkeypatch.setitem(sys.modules, 'nltk', SimpleNamespace(data=data))


def test_check_nltk_data_skips_existing_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _FakeNltkData(available=True)
    _install_fake_nltk(monkeypatch, data)
    download_calls = []
    monkeypatch.setattr(resource_utils, 'download_url', lambda *args, **kwargs: download_calls.append((args, kwargs)))

    resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert data.lookups == ['taggers/averaged_perceptron_tagger_eng/']
    assert download_calls == []


def test_check_nltk_data_downloads_english_tagger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))
    download_urls = []

    def fake_download(url: str, save_path: str) -> None:
        download_urls.append(url)
        with zipfile.ZipFile(save_path, 'w') as archive:
            archive.writestr('averaged_perceptron_tagger_eng/weights.json', '{}')
        data.available = True

    monkeypatch.setattr(resource_utils, 'download_url', fake_download)

    resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert download_urls == [
        'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/'
        'averaged_perceptron_tagger_eng.zip'
    ]
    assert data.lookups == [
        'taggers/averaged_perceptron_tagger_eng/',
        'taggers/averaged_perceptron_tagger_eng/',
    ]
    assert not (tmp_path / 'nltk_data/taggers/averaged_perceptron_tagger_eng.zip').exists()


def test_check_nltk_data_raises_when_resource_is_still_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))

    def fake_download(url: str, save_path: str) -> None:
        with zipfile.ZipFile(save_path, 'w') as archive:
            archive.writestr('wrong_resource/weights.json', '{}')

    monkeypatch.setattr(resource_utils, 'download_url', fake_download)

    with pytest.raises(RuntimeError, match='still unavailable after download'):
        resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert not (tmp_path / 'nltk_data/taggers/averaged_perceptron_tagger_eng.zip').exists()


def test_check_nltk_data_propagates_download_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))

    def fail_download(url: str, save_path: str) -> None:
        raise OSError('offline')

    monkeypatch.setattr(resource_utils, 'download_url', fail_download)

    with pytest.raises(RuntimeError, match='All mirrors failed'):
        resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert not (tmp_path / 'nltk_data/taggers/averaged_perceptron_tagger_eng.zip').exists()
