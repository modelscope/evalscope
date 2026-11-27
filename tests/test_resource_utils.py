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
    """Resources installed under a permissive umask must be usable by newer NLTK."""
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))

    def fake_download(url: str, save_path: str, sha256: Optional[str] = None) -> None:
        with zipfile.ZipFile(save_path, 'w') as archive:
            archive.writestr('averaged_perceptron_tagger_eng/weights.json', '{}')
        data.available = True

    monkeypatch.setattr(resource_utils, 'download_url', fake_download)

    previous_umask = os.umask(0o002)
    try:
        resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')
    finally:
        os.umask(previous_umask)

    base = tmp_path / 'nltk_data'
    assert (base / 'taggers/averaged_perceptron_tagger_eng/weights.json').is_file()
    for path in [base, *base.rglob('*')]:
        assert not path.stat().st_mode & 0o022, f'{path} stayed writable by group or other'


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_make_owner_private_resolves_entries_relative_to_the_walked_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Walked names must resolve under the resource, not the working directory."""
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
    """Neither child links nor a linked walk root may change external permissions."""
    resource = tmp_path / 'data'
    resource.mkdir()
    outside_file = tmp_path / 'outside.txt'
    outside_file.write_text('an unrelated shared file')
    outside_file.chmod(0o664)
    outside_dir = tmp_path / 'outside_dir'
    outside_dir.mkdir()
    outside_dir.chmod(0o775)
    outside_child = outside_dir / 'shared.txt'
    outside_child.write_text('an unrelated shared file')
    outside_child.chmod(0o664)
    root_link = tmp_path / 'linked_resource'
    try:
        (resource / 'linked.txt').symlink_to(outside_file)
        (resource / 'linked_dir').symlink_to(outside_dir, target_is_directory=True)
        root_link.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform without symlink support
        pytest.skip('this platform cannot create symlinks')

    resource_utils._make_owner_private(str(resource))
    resource_utils._make_owner_private(str(root_link))

    assert _mode(outside_file) == 0o664, 'a file symlink target outside the resource was modified'
    assert _mode(outside_dir) == 0o775, 'a directory symlink target outside the resource was modified'
    assert _mode(outside_child) == 0o664, 'a linked walk root was traversed'
    assert (resource / 'linked.txt').is_symlink(), 'the symlink itself was replaced or removed'


@pytest.mark.parametrize('linked_path', ['nltk_data', 'nltk_data/taggers', 'nltk_data/taggers/averaged_perceptron_tagger_eng'])
def test_check_nltk_data_rejects_linked_install_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, linked_path: str
) -> None:
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))
    outside = tmp_path / 'outside'
    outside.mkdir()
    outside.chmod(0o775)
    shared = outside / 'shared.txt'
    shared.write_text('unrelated')
    shared.chmod(0o664)
    link = tmp_path / linked_path
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('this platform cannot create symlinks')

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail('a linked installation path must be rejected before downloading')

    monkeypatch.setattr(resource_utils, 'download_url', unexpected_download)
    with pytest.raises(RuntimeError, match='through a symlink'):
        resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert _mode(outside) == 0o775
    assert _mode(shared) == 0o664
    assert list(outside.iterdir()) == [shared]


def test_failed_download_preserves_existing_directory_permissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    monkeypatch.setenv('HOME', str(tmp_path))
    extract_dir = tmp_path / 'nltk_data' / 'taggers'
    extract_dir.mkdir(parents=True)
    extract_dir.chmod(0o775)
    extract_dir.parent.chmod(0o775)

    def fail_download(*args: object, **kwargs: object) -> None:
        raise OSError('offline')

    monkeypatch.setattr(resource_utils, 'download_url', fail_download)
    with pytest.raises(RuntimeError, match='All mirrors failed'):
        resource_utils.check_nltk_data('averaged_perceptron_tagger_eng')

    assert _mode(extract_dir) == 0o775
    assert _mode(extract_dir.parent) == 0o775


def test_check_nltk_data_leaves_neighbouring_resources_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Other resources and the home directory retain their permissions."""
    data = _FakeNltkData()
    _install_fake_nltk(monkeypatch, data)
    home = tmp_path / 'home'
    home.mkdir()
    home.chmod(0o775)
    monkeypatch.setenv('HOME', str(home))

    neighbouring = home / 'nltk_data' / 'taggers' / 'other_tagger'
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
