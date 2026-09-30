import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest
from datasets import Dataset as HFDataset

from evalscope.api.dataset import DatasetHub, download_dataset_file, download_dataset_snapshot, load_dataset_from_hub
from evalscope.constants import HubType


def _install_modelscope_loaders(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_download: Callable[..., str],
    dataset_load: Callable[..., HFDataset],
) -> None:
    fake_modelscope = types.ModuleType('modelscope')
    fake_modelscope.snapshot_download = snapshot_download
    fake_modelscope.MsDataset = types.SimpleNamespace(load=dataset_load)
    fake_constants = types.ModuleType('modelscope.utils.constant')
    fake_constants.DownloadMode = types.SimpleNamespace(FORCE_REDOWNLOAD='force_redownload')
    monkeypatch.setitem(sys.modules, 'modelscope', fake_modelscope)
    monkeypatch.setitem(sys.modules, 'modelscope.utils.constant', fake_constants)


def _write_snapshot(snapshot: Path) -> None:
    snapshot.mkdir(parents=True)
    (snapshot / 'test.jsonl').write_text(json.dumps({'text': 'cached-test'}) + '\n', encoding='utf-8')
    (snapshot / 'train.jsonl').write_text(json.dumps({'text': 'cached-train'}) + '\n', encoding='utf-8')


def test_modelscope_load_prefers_cached_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / 'datasets' / 'owner--data' / 'snapshots' / 'master'
    _write_snapshot(snapshot)
    calls = []

    def cached_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('A cached snapshot must not use MsDataset.load')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    dataset = DatasetHub(data_id_or_path='owner/data').load(split='test')

    assert dataset['text'] == ['cached-test']
    assert calls == [{'repo_id': 'owner/data', 'repo_type': 'dataset', 'local_files_only': True}]


def test_modelscope_cache_miss_keeps_remote_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def missing_snapshot(**kwargs: Any) -> str:
        raise ValueError('No cached snapshot')

    def remote_load(**kwargs: Any) -> HFDataset:
        calls.update(kwargs)
        return HFDataset.from_dict({'text': ['remote']})

    _install_modelscope_loaders(monkeypatch, missing_snapshot, remote_load)

    dataset = load_dataset_from_hub('owner/data', split='test', subset='main', version='v2')

    assert dataset['text'] == ['remote']
    assert calls == {
        'dataset_name': 'owner/data', 'split': 'test', 'subset_name': 'main',
        'trust_remote_code': True, 'version': 'v2',
    }


def test_modelscope_force_redownload_skips_snapshot_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def cached_snapshot(**kwargs: Any) -> str:
        pytest.fail('force_redownload must skip the offline cache probe')

    def remote_load(**kwargs: Any) -> HFDataset:
        calls.update(kwargs)
        return HFDataset.from_dict({'text': ['fresh']})

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    dataset = load_dataset_from_hub('owner/data', split='test', force_redownload=True)

    assert dataset['text'] == ['fresh']
    assert calls['download_mode'] == 'force_redownload'


@pytest.mark.parametrize('source', [HubType.LOCAL, HubType.MODELSCOPE])
def test_explicit_local_dataset_skips_modelscope_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    snapshot = tmp_path / 'explicit'
    _write_snapshot(snapshot)

    def cached_snapshot(**kwargs: Any) -> str:
        pytest.fail('An explicit local path must not probe the ModelScope cache')

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('An explicit local path must not load remotely')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    dataset = load_dataset_from_hub(str(snapshot), split='test', data_source=source)

    assert dataset['text'] == ['cached-test']


@pytest.mark.parametrize('cached_revision', ['v2', 'master', None])
def test_modelscope_pinned_revision_does_not_reuse_other_or_unversioned_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached_revision: str | None,
) -> None:
    snapshot = tmp_path / 'snapshots' / cached_revision if cached_revision else tmp_path / 'legacy'
    _write_snapshot(snapshot)
    calls = []

    def cached_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        return HFDataset.from_dict({'text': ['remote-v2']})

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    dataset = load_dataset_from_hub('owner/data', split='test', version='v2')

    assert dataset['text'] == (['cached-test'] if cached_revision == 'v2' else ['remote-v2'])
    assert calls[0]['revision'] == 'v2'


def test_modelscope_cached_data_errors_do_not_trigger_remote_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()

    def cached_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('A cached dataset parsing error must not silently download another dataset')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    with pytest.raises(FileNotFoundError):
        load_dataset_from_hub('owner/data', split='test')


def test_modelscope_cached_snapshot_keeps_repository_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / 'snapshot'
    _write_snapshot(snapshot)
    metadata = snapshot / 'dataset_infos.json'
    metadata.write_text('{}', encoding='utf-8')

    def cached_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('A cached snapshot must load locally')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    dataset = load_dataset_from_hub('owner/data', split='test')

    assert dataset['text'] == ['cached-test']
    assert metadata.read_text(encoding='utf-8') == '{}'


def test_huggingface_load_does_not_probe_modelscope_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    calls = {}

    def cached_snapshot(**kwargs: Any) -> str:
        pytest.fail('Hugging Face must not probe the ModelScope cache')

    def dataset_load(**kwargs: Any) -> HFDataset:
        calls.update(kwargs)
        return HFDataset.from_dict({'text': ['huggingface']})

    _install_modelscope_loaders(monkeypatch, cached_snapshot, dataset_load)
    monkeypatch.setattr(datasets, 'load_dataset', dataset_load)

    dataset = load_dataset_from_hub('owner/data', split='test', data_source=HubType.HUGGINGFACE)

    assert dataset['text'] == ['huggingface']
    assert calls['path'] == 'owner/data'


def test_modelscope_partial_cache_does_not_skip_requested_snapshot_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / 'datasets' / 'owner--data' / 'snapshots' / 'master'
    snapshot.mkdir(parents=True)
    (snapshot / 'README.md').write_text('A partially downloaded dataset.', encoding='utf-8')
    calls = []
    fake_modelscope = types.ModuleType('modelscope')

    def partial_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def complete_snapshot(dataset_id: str, **kwargs: Any) -> str:
        calls.append((dataset_id, kwargs))
        (snapshot / 'required.jsonl').write_text('{}\n', encoding='utf-8')
        return str(snapshot)

    fake_modelscope.snapshot_download = partial_snapshot
    fake_modelscope.dataset_snapshot_download = complete_snapshot
    monkeypatch.setitem(sys.modules, 'modelscope', fake_modelscope)

    result = download_dataset_snapshot('owner/data', allow_file_pattern=['required.jsonl'])

    assert result == str(snapshot)
    assert (snapshot / 'required.jsonl').is_file()
    assert calls == [('owner/data', {'allow_file_pattern': ['required.jsonl']})]


@pytest.mark.parametrize('cache_layout', ['snapshot', 'legacy'])
def test_native_gsm8k_uses_sdk_cache_without_dataset_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_layout: str,
) -> None:
    pytest.importorskip('modelscope_hub')
    import httpx
    import requests
    from modelscope.hub.snapshot_download import snapshot_download
    from modelscope_hub import config as hub_config

    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    cache_root = tmp_path / 'modelscope'
    if cache_layout == 'snapshot':
        snapshot = cache_root / 'datasets' / 'AI-ModelScope--gsm8k' / 'snapshots' / 'master'
    else:
        snapshot = cache_root / 'datasets' / 'AI-ModelScope' / 'gsm8k'
    snapshot.mkdir(parents=True)
    records = '\n'.join(json.dumps({'question': 'What is 1+1?', 'answer': '1+1=2 #### 2'}) for _ in range(4))
    (snapshot / 'test.jsonl').write_text(records + '\n', encoding='utf-8')
    (snapshot / 'train.jsonl').write_text(records + '\n', encoding='utf-8')
    (snapshot / 'README.md').write_text(
        '---\nconfigs:\n- config_name: main\n  data_files:\n'
        '  - split: train\n    path: train.jsonl\n  - split: test\n    path: test.jsonl\n---\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('MODELSCOPE_CACHE', str(cache_root))
    monkeypatch.setattr(hub_config, '_default_config', None)

    def forbid_network(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Loading a pre-downloaded Native dataset must not access the network')

    monkeypatch.setattr(requests.sessions.Session, 'request', forbid_network)
    monkeypatch.setattr(httpx.Client, 'request', forbid_network)
    _install_modelscope_loaders(monkeypatch, snapshot_download, forbid_network)
    config = TaskConfig(datasets=['gsm8k'], limit=1, dataset_dir=str(tmp_path / 'evalscope'))
    benchmark = get_benchmark('gsm8k', config)

    dataset = benchmark.load_dataset()

    assert benchmark.dataset_id == 'AI-ModelScope/gsm8k'
    assert list(dataset.keys()) == ['main']
    assert len(dataset['main']) == 1
    assert dataset['main'][0].target == '2'
    assert len(benchmark.fewshot_dataset['main']) == 4


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


def test_download_file_modelscope_uses_single_file_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    image_path = tmp_path / 'images' / 'page_[1].png'
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b'image')
    fake_modelscope = types.ModuleType('modelscope')

    def fake_download(dataset_id: str, file_path: str, **kwargs: Any) -> str:
        calls.append((dataset_id, file_path, kwargs))
        if kwargs.get('local_files_only'):
            raise ValueError('cache miss')
        return str(image_path)

    fake_modelscope.dataset_file_download = fake_download
    monkeypatch.setitem(sys.modules, 'modelscope', fake_modelscope)

    result = download_dataset_file(
        'remote-dataset',
        'images/page_[1].png',
        data_source=HubType.MODELSCOPE,
        revision='v1',
    )

    assert result == str(image_path)
    assert calls == [
        ('remote-dataset', 'images/page_[1].png', {'revision': 'v1', 'local_files_only': True}),
        ('remote-dataset', 'images/page_[1].png', {'revision': 'v1'}),
    ]
