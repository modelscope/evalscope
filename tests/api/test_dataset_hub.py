import sys
import types

import pytest

from evalscope.api.dataset import DatasetHub, download_dataset_file, download_dataset_snapshot
from evalscope.constants import HubType


def test_download_snapshot_resolves_existing_local_path(tmp_path) -> None:
    snapshot_dir = tmp_path / 'dataset'
    snapshot_dir.mkdir()

    assert download_dataset_snapshot(str(snapshot_dir)) == str(snapshot_dir.resolve())


def test_download_snapshot_modelscope_passes_file_patterns(monkeypatch) -> None:
    calls = {}
    fake_modelscope = types.ModuleType('modelscope')

    def fake_download(dataset_id, **kwargs):
        calls['dataset_id'] = dataset_id
        calls['kwargs'] = kwargs
        return '/tmp/modelscope_snapshot'

    fake_modelscope.dataset_snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, 'modelscope', fake_modelscope)

    result = download_dataset_snapshot(
        'remote-dataset',
        data_source=HubType.MODELSCOPE,
        revision='v1',
        cache_dir='/tmp/cache',
        allow_file_pattern=['data.jsonl'],
        ignore_file_pattern=['unused/*'],
    )

    assert result == '/tmp/modelscope_snapshot'
    assert calls == {
        'dataset_id': 'remote-dataset',
        'kwargs': {
            'revision': 'v1',
            'cache_dir': '/tmp/cache',
            'allow_file_pattern': ['data.jsonl'],
            'ignore_file_pattern': ['unused/*'],
        },
    }


def test_download_snapshot_huggingface_uses_dataset_repo(monkeypatch) -> None:
    calls = {}
    fake_huggingface_hub = types.ModuleType('huggingface_hub')

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return '/tmp/hf_snapshot'

    fake_huggingface_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_huggingface_hub)

    hub = DatasetHub(
        data_id_or_path='org/data',
        data_source=HubType.HUGGINGFACE,
        revision='main',
        force_redownload=True,
        cache_dir='/tmp/cache',
    )

    assert hub.download_snapshot(allow_file_pattern='data.jsonl') == '/tmp/hf_snapshot'
    assert calls == {
        'repo_id': 'org/data',
        'repo_type': 'dataset',
        'revision': 'main',
        'cache_dir': '/tmp/cache',
        'force_download': True,
        'allow_patterns': 'data.jsonl',
        'ignore_patterns': None,
    }


def test_download_file_keeps_local_path_traversal_protection(tmp_path) -> None:
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()

    with pytest.raises(ValueError):
        download_dataset_file(str(dataset_dir), '../secret.jsonl', data_source=HubType.LOCAL)
