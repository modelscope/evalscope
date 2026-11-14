import os
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

import pytest

from evalscope.utils import resource_utils
from evalscope.utils.resource_utils import MIRROR_MAP


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
    download_calls = []

    def fake_download(url: str, save_path: str, sha256: Optional[str] = None) -> None:
        download_calls.append({'url': url, 'sha256': sha256})
        with zipfile.ZipFile(save_path, 'w') as archive:
            archive.writestr('averaged_perceptron_tagger_eng/weights.json', '{}')
        data.available = True

    monkeypatch.setattr(resource_utils, 'download_url', fake_download)

    resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    pinned = MIRROR_MAP['averaged_perceptron_tagger_eng']['mirrors'][0]
    assert download_calls == [pinned]
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

    def fake_download(url: str, save_path: str, sha256: Optional[str] = None) -> None:
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

    def fail_download(url: str, save_path: str, sha256: Optional[str] = None) -> None:
        raise OSError('offline')

    monkeypatch.setattr(resource_utils, 'download_url', fail_download)

    with pytest.raises(RuntimeError, match='All mirrors failed'):
        resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert not (tmp_path / 'nltk_data/taggers/averaged_perceptron_tagger_eng.zip').exists()


def test_mirror_map_pins_valid_sha256_digests() -> None:
    """Every enabled mirror archive must use a valid pinned SHA-256 digest."""
    for meta in MIRROR_MAP.values():
        for mirror in meta['mirrors']:
            assert re.fullmatch(r'[0-9a-f]{64}', mirror['sha256']), mirror['url']


def test_check_nltk_data_does_not_extract_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))
    download_calls = []

    def fail_checksum(url: str, save_path: str, sha256: Optional[str] = None) -> None:
        download_calls.append({'url': url, 'sha256': sha256})
        raise ValueError(f'Checksum mismatch for {url}')

    def unexpected_extract(*args: object, **kwargs: object) -> None:
        raise AssertionError('Checksum-mismatched archive must not be extracted')

    monkeypatch.setattr(resource_utils, 'download_url', fail_checksum)
    monkeypatch.setattr(resource_utils.zipfile, 'ZipFile', unexpected_extract)

    with pytest.raises(RuntimeError, match='All mirrors failed'):
        resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    pinned = MIRROR_MAP['averaged_perceptron_tagger_eng']['mirrors'][0]
    assert download_calls == [pinned]
    assert not (tmp_path / 'nltk_data/taggers/averaged_perceptron_tagger_eng.zip').exists()


def test_check_nltk_data_makes_downloaded_directories_owner_private(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A group-writable ``~/nltk_data`` makes NLTK ignore a download that succeeded.

    ``os.makedirs`` honours the process umask, and ``ZipInfo`` keeps the mode recorded when the archive was built, so
    on a host running with ``umask 0002`` both the created directories and the extracted ones end up group-writable.
    NLTK then refuses to read them, and the caller reports "still unavailable after download" for a healthy archive.
    """
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))

    def fake_download(url: str, save_path: str, sha256: Optional[str] = None) -> None:
        with zipfile.ZipFile(save_path, 'w') as archive:
            archive.writestr('averaged_perceptron_tagger_eng/weights.json', '{}')
        data.available = True

    monkeypatch.setattr(resource_utils, 'download_url', fake_download)

    # Simulate a shared host: every directory this call creates starts group-writable.
    original_makedirs = os.makedirs

    def group_writable_makedirs(name, mode=0o777, exist_ok=False):
        original_makedirs(name, mode=mode, exist_ok=exist_ok)
        os.chmod(name, os.stat(name).st_mode | 0o022)

    monkeypatch.setattr(resource_utils.os, 'makedirs', group_writable_makedirs)

    resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    directories = [path for path in (tmp_path / 'nltk_data').rglob('*') if path.is_dir()]
    assert directories, 'the download should have created the resource directory tree'
    for path in directories:
        assert not path.stat().st_mode & 0o022, f'{path} stayed writable by group or other'


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_make_owner_private_resolves_entries_relative_to_the_walked_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``os.walk`` reports bare names in ``dirnames``; those must be joined to the walked root, not left bare.

    A bare name handed to ``chmod`` resolves against the process working directory, so running the helper from a
    directory that happens to contain a same-named folder used to change that unrelated folder's permissions too.
    The helper runs before the download, so even a download that then fails left the side effect behind.
    """
    resource = tmp_path / 'data'
    (resource / 'tokenizers').mkdir(parents=True)
    unrelated = tmp_path / 'work' / 'tokenizers'
    unrelated.mkdir(parents=True)
    unrelated.chmod(0o775)
    monkeypatch.chdir(unrelated.parent)

    resource_utils._make_owner_private(str(resource))

    assert _mode(unrelated) == 0o775, 'a same-named folder under the working directory was modified'
    assert not _mode(resource / 'tokenizers') & 0o022


def test_make_owner_private_does_not_follow_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``os.walk`` does not descend directory links, but ``os.stat`` / ``os.chmod`` still resolve both link kinds.

    Tightening a symlink therefore tightens whatever it points at, which can be a file outside the resource being
    installed.
    """
    resource = tmp_path / 'data'
    resource.mkdir()
    outside_file = tmp_path / 'outside.txt'
    outside_file.write_text('an unrelated shared file')
    outside_file.chmod(0o664)
    outside_dir = tmp_path / 'outside_dir'
    outside_dir.mkdir()
    outside_dir.chmod(0o775)
    try:
        (resource / 'linked.txt').symlink_to(outside_file)
        (resource / 'linked_dir').symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform without symlink support
        pytest.skip('this platform cannot create symlinks')

    resource_utils._make_owner_private(str(resource))

    assert _mode(outside_file) == 0o664, 'a file symlink target outside the resource was modified'
    assert _mode(outside_dir) == 0o775, 'a directory symlink target outside the resource was modified'
    assert (resource / 'linked.txt').is_symlink(), 'the symlink itself was replaced or removed'


def test_tighten_up_to_stops_at_a_stop_that_is_not_an_ancestor(tmp_path: Path) -> None:
    """A ``stop`` outside the chain must not send the walk climbing into directories nobody named."""
    chain_root = tmp_path / 'chain' / 'taggers' / 'tagger'
    chain_root.mkdir(parents=True)
    chain_root.chmod(0o775)
    (tmp_path / 'chain' / 'taggers').chmod(0o775)
    unrelated_stop = tmp_path / 'elsewhere'
    unrelated_stop.mkdir()
    unrelated_stop.chmod(0o775)

    resource_utils._tighten_up_to(str(chain_root), str(unrelated_stop))

    assert not _mode(chain_root) & 0o022, 'the target itself was not tightened'
    assert _mode(tmp_path / 'chain' / 'taggers') == 0o775, 'the chain climbed past its start'
    assert _mode(unrelated_stop) == 0o775


def test_check_nltk_data_leaves_neighbouring_resources_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only the resource being installed is tightened, not everything already under ``~/nltk_data``.

    ``corpora`` holds several resources; installing the tagger must not rewrite the permissions its owner set on an
    existing corpus, nor the home directory above the data directory.
    """
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    home = tmp_path / 'home'
    home.mkdir()
    home.chmod(0o775)
    monkeypatch.setenv('HOME', str(home))

    neighbouring = home / 'nltk_data' / 'corpora' / 'stopwords'
    neighbouring.mkdir(parents=True)
    neighbouring.chmod(0o775)
    (neighbouring / 'english').write_text('a shared corpus file')
    (neighbouring / 'english').chmod(0o664)

    def fake_download(url: str, save_path: str, sha256: Optional[str] = None) -> None:
        with zipfile.ZipFile(save_path, 'w') as archive:
            archive.writestr('averaged_perceptron_tagger_eng/weights.json', '{}')
        data.available = True

    monkeypatch.setattr(resource_utils, 'download_url', fake_download)

    resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert _mode(neighbouring) == 0o775, 'an already-present resource was tightened'
    assert _mode(neighbouring / 'english') == 0o664
    assert _mode(home) == 0o775, 'the home directory above the data directory was tightened'
    assert not _mode(home / 'nltk_data' / 'taggers' / 'averaged_perceptron_tagger_eng') & 0o022
