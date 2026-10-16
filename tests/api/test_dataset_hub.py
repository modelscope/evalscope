import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest
from datasets import Dataset as HFDataset

from evalscope.api.dataset import (
    DatasetHub,
    DatasetSource,
    download_dataset_file,
    download_dataset_snapshot,
    load_dataset_from_hub,
)
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


def test_dataset_source_resolves_hub_specific_location() -> None:
    source = DatasetSource(
        dataset_id='ms-org/data',
        dataset_revision='ms-revision',
        hf_dataset_id='hf-org/data',
        hf_dataset_revision='hf-revision',
        hf_file_paths={'data.jsonl': 'files/data.jsonl'},
    )

    assert source.resolve_dataset_id(HubType.MODELSCOPE) == 'ms-org/data'
    assert source.resolve_revision(HubType.MODELSCOPE) == 'ms-revision'
    assert source.resolve_file_path(HubType.MODELSCOPE, 'data.jsonl') == 'data.jsonl'
    assert source.resolve_dataset_id(HubType.HUGGINGFACE) == 'hf-org/data'
    assert source.resolve_revision(HubType.HUGGINGFACE) == 'hf-revision'
    assert source.resolve_file_path(HubType.HUGGINGFACE, 'data.jsonl') == 'files/data.jsonl'


def test_dataset_source_preserves_existing_local_path(tmp_path: Path) -> None:
    local_path = tmp_path / 'dataset'
    local_path.mkdir()
    source = DatasetSource(dataset_id=str(local_path), hf_dataset_id='hf-org/data')

    assert source.is_local
    assert source.resolve_dataset_id(HubType.HUGGINGFACE) == str(local_path)


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


@pytest.mark.parametrize(
    'options',
    [
        {'use_streaming': False},
        {'target': 'text'},
        {'download_mode': 'reuse_dataset_if_exists'},
        {'download_mode': 'force_redownload'},
    ],
)
def test_modelscope_specific_options_keep_sdk_loading(
    options: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_snapshot(tmp_path / 'snapshot')
    calls = []

    def forbid_probe(**kwargs: Any) -> str:
        pytest.fail('ModelScope controls must retain the SDK loader even when a snapshot exists')

    def sdk_load(**kwargs: Any) -> HFDataset:
        calls.append(kwargs)
        return HFDataset.from_dict({'text': ['sdk']})

    _install_modelscope_loaders(monkeypatch, forbid_probe, sdk_load)
    with load_dataset_from_hub('owner/data', split='test', subset='main', version='v2', **options) as dataset:
        assert dataset['text'] == ['sdk']
    assert calls == [{
        'dataset_name': 'owner/data', 'split': 'test', 'subset_name': 'main',
        'version': 'v2', 'trust_remote_code': True, **options,
    }]


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
    (snapshot / 'test.jsonl').write_text('not valid JSON\n', encoding='utf-8')

    def cached_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('A cached dataset parsing error must not silently download another dataset')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    from datasets.exceptions import DatasetGenerationError

    with pytest.raises(DatasetGenerationError):
        load_dataset_from_hub('owner/data', split='test')


@pytest.mark.parametrize('cache_layout', ['snapshot', 'legacy'])
@pytest.mark.parametrize('missing_content', ['readme_only', 'declared_file', 'split', 'sliced_split', 'streaming_split'])
def test_modelscope_partial_split_cache_falls_back_to_remote_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_layout: str, missing_content: str,
) -> None:
    pytest.importorskip('modelscope_hub')
    import httpx
    import requests
    from modelscope.hub.snapshot_download import snapshot_download
    from modelscope_hub import config as hub_config

    cache_root = tmp_path / 'modelscope'
    repository = 'owner--data/snapshots/master' if cache_layout == 'snapshot' else 'owner/data'
    snapshot = cache_root / 'datasets' / repository
    snapshot.mkdir(parents=True)
    (snapshot / 'README.md').write_text('A partial dataset repository.', encoding='utf-8')
    if missing_content != 'readme_only':
        (snapshot / 'train.jsonl').write_text('{"text": "cached-train"}\n', encoding='utf-8')
    if missing_content == 'declared_file':
        (snapshot / 'README.md').write_text(
            '---\nconfigs:\n- config_name: default\n  data_files:\n'
            '  - split: train\n    path: train.jsonl\n  - split: test\n    path: absent.jsonl\n---\n',
            encoding='utf-8',
        )
    monkeypatch.setenv('MODELSCOPE_CACHE', str(cache_root))
    monkeypatch.setattr(hub_config, '_default_config', None)

    def forbid_network(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('The cache probe and local split loading must not make HTTP requests')

    requested_split = 'test[:1]' if missing_content == 'sliced_split' else 'test'
    streaming_kwargs = {'streaming': True} if missing_content == 'streaming_split' else {}

    def remote_load(**kwargs: Any) -> HFDataset:
        assert kwargs == {
            'dataset_name': 'owner/data', 'split': requested_split, 'subset_name': None,
            'trust_remote_code': True, 'data_dir': '.', **streaming_kwargs,
        }
        return HFDataset.from_dict({'text': ['remote-test']})

    monkeypatch.setattr(requests.sessions.Session, 'request', forbid_network)
    monkeypatch.setattr(httpx.Client, 'request', forbid_network)
    _install_modelscope_loaders(monkeypatch, snapshot_download, remote_load)

    dataset = load_dataset_from_hub('owner/data', split=requested_split, data_dir='.', **streaming_kwargs)

    assert dataset['text'] == ['remote-test']


@pytest.mark.parametrize('source', [HubType.LOCAL, HubType.MODELSCOPE])
def test_explicit_partial_local_dataset_does_not_fall_back_to_remote_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    from datasets.data_files import EmptyDatasetError

    snapshot = tmp_path / 'explicit'
    snapshot.mkdir()
    (snapshot / 'README.md').write_text('A partial local dataset.', encoding='utf-8')

    def forbid_remote(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('An explicit local path must not fall back to a remote source')

    _install_modelscope_loaders(monkeypatch, forbid_remote, forbid_remote)

    with pytest.raises(EmptyDatasetError):
        load_dataset_from_hub(str(snapshot), split='test', data_source=source)


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


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('indices', [(0,), (1,), (0, 2), (0, 1, 2)])
def test_modelscope_cached_numbered_shards_preserve_the_complete_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indices: tuple[int, ...], streaming: bool,
) -> None:
    snapshot = tmp_path / 'snapshots' / 'master'
    snapshot.mkdir(parents=True)
    (snapshot / 'README.md').write_text(
        '---\nconfigs:\n- config_name: default\n  data_files:\n'
        '  - split: test\n    path: test-*.jsonl\n---\n', encoding='utf-8',
    )
    for index in indices:
        (snapshot / f'test-{index:05d}-of-00003.jsonl').write_text(
            json.dumps({'text': f'row-{index}'}) + '\n', encoding='utf-8',
        )
    remote_calls = []
    cache_dir = str(tmp_path / 'hf-cache')
    requested_split = 'test' if streaming else 'test[:3]'

    def cached_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        remote_calls.append(kwargs)
        assert kwargs == {
            'dataset_name': 'owner/data', 'split': requested_split, 'subset_name': None,
            'trust_remote_code': True, 'cache_dir': cache_dir, 'streaming': streaming,
        }
        return HFDataset.from_dict({'text': ['row-0', 'row-1', 'row-2']})

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)
    dataset = load_dataset_from_hub('owner/data', split=requested_split, cache_dir=cache_dir, streaming=streaming)

    assert [row['text'] for row in dataset] == ['row-0', 'row-1', 'row-2']
    assert len(remote_calls) == (0 if len(indices) == 3 else 1)


def test_modelscope_complete_snapshots_do_not_share_arrow_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = {}
    for name in ['first', 'second']:
        snapshot = tmp_path / name / 'snapshots' / 'master'
        _write_snapshot(snapshot)
        (snapshot / 'test.jsonl').write_text(json.dumps({'text': name}) + '\n', encoding='utf-8')
        (snapshot / 'README.md').write_text(
            '---\nconfigs:\n- config_name: default\n  data_files:\n'
            '  - split: test\n    path: test.jsonl\n---\n', encoding='utf-8',
        )
        snapshots[f'owner/{name}'] = str(snapshot)

    def cached_snapshot(**kwargs: Any) -> str:
        return snapshots[kwargs['repo_id']]

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('Both complete snapshots must load locally')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)
    cache_dir = str(tmp_path / 'shared-hf-cache')
    first = load_dataset_from_hub('owner/first', split='test', cache_dir=cache_dir)
    second = load_dataset_from_hub('owner/second', split='test', cache_dir=cache_dir)
    first_again = load_dataset_from_hub('owner/first', split='test', cache_dir=cache_dir)

    assert first['text'] == ['first']
    assert second['text'] == ['second']
    assert first_again['text'] == ['first']
    assert first.cache_files != second.cache_files
    assert first.cache_files == first_again.cache_files


def test_modelscope_cached_loading_preserves_loader_and_builder_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'test.jsonl').write_text(
        json.dumps({'records': [{'text': 'cached'}]}) + '\n', encoding='utf-8',
    )

    def cached_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('The complete snapshot must support both loader and builder options')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)
    dataset = load_dataset_from_hub(
        'owner/data', split='test', cache_dir=str(tmp_path / 'hf-cache'),
        keep_in_memory=True, num_proc=1, save_infos=False, verification_mode='no_checks',
        field='records', writer_batch_size=1,
    )

    assert dataset['text'] == ['cached']
    assert dataset.cache_files == []


def test_modelscope_explicit_data_files_preserve_partial_shard_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    selected_file = snapshot / 'test-00000-of-00002.jsonl'
    selected_file.write_text('{"text": "selected"}\n', encoding='utf-8')

    def cached_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('An explicit file selection must not require the other shards')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)
    dataset = load_dataset_from_hub(
        'owner/data', split='test', data_files={'test': str(selected_file)}, cache_dir=str(tmp_path / 'hf-cache'),
    )

    assert dataset['text'] == ['selected']


@pytest.mark.parametrize('filenames', [
    ['test-00000-of-00000.jsonl'],
    ['test-00002-of-00002.jsonl'],
    ['test-00000-of-00002.jsonl', 'test-00001-of-00003.jsonl'],
    ['test-0-of-2.jsonl', 'test-00000-of-00002.jsonl'],
])
def test_modelscope_invalid_numbered_shards_do_not_hide_data_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filenames: list[str],
) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    for filename in filenames:
        (snapshot / filename).write_text('{"text": "cached"}\n', encoding='utf-8')

    def cached_snapshot(**kwargs: Any) -> str:
        return str(snapshot)

    def remote_load(**kwargs: Any) -> HFDataset:
        pytest.fail('Invalid shard numbering must not be treated as missing data')

    _install_modelscope_loaders(monkeypatch, cached_snapshot, remote_load)

    with pytest.raises(ValueError, match='numbered shard'):
        load_dataset_from_hub('owner/data', split='test', cache_dir=str(tmp_path / 'hf-cache'))


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
@pytest.mark.parametrize('missing_shard', [False, True])
def test_native_gsm8k_uses_sdk_cache_without_dataset_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_layout: str, missing_shard: bool,
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
    test_file = 'test-00000-of-00002.jsonl' if missing_shard else 'test.jsonl'
    test_records = records.splitlines()[0] if missing_shard else records
    (snapshot / test_file).write_text(test_records + '\n', encoding='utf-8')
    (snapshot / 'train.jsonl').write_text(records + '\n', encoding='utf-8')
    test_pattern = 'test-*.jsonl' if missing_shard else 'test.jsonl'
    (snapshot / 'README.md').write_text(
        '---\nconfigs:\n- config_name: main\n  data_files:\n'
        f'  - split: train\n    path: train.jsonl\n  - split: test\n    path: {test_pattern}\n---\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('MODELSCOPE_CACHE', str(cache_root))
    monkeypatch.setattr(hub_config, '_default_config', None)

    def forbid_network(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Loading a pre-downloaded Native dataset must not access the network')

    remote_calls = []

    def remote_load(**kwargs: Any) -> HFDataset:
        remote_calls.append(kwargs)
        assert missing_shard
        assert kwargs['split'] in ['test', 'train']
        assert kwargs == {
            'dataset_name': 'AI-ModelScope/gsm8k', 'split': kwargs['split'], 'subset_name': 'main',
            'trust_remote_code': True,
        }
        return HFDataset.from_list([json.loads(record) for record in records.splitlines()])

    monkeypatch.setattr(requests.sessions.Session, 'request', forbid_network)
    monkeypatch.setattr(httpx.Client, 'request', forbid_network)
    _install_modelscope_loaders(monkeypatch, snapshot_download, remote_load)
    config = TaskConfig(datasets=['gsm8k'], limit=None if missing_shard else 1, dataset_dir=str(tmp_path / 'evalscope'))
    benchmark = get_benchmark('gsm8k', config)

    dataset = benchmark.load_dataset()

    assert benchmark.dataset_id == 'AI-ModelScope/gsm8k'
    assert list(dataset.keys()) == ['main']
    assert len(dataset['main']) == (4 if missing_shard else 1)
    assert dataset['main'][0].target == '2'
    assert len(benchmark.fewshot_dataset['main']) == 4
    assert len(remote_calls) == (2 if missing_shard else 0)


def test_benchmark_dataset_source_defaults_to_modelscope_without_task_config() -> None:
    from evalscope.api.registry import get_benchmark

    benchmark = get_benchmark('gsm8k')

    assert benchmark.dataset_hub == HubType.MODELSCOPE
    assert benchmark.dataset_id == 'AI-ModelScope/gsm8k'
    assert benchmark.dataset_revision is None


def test_explicit_dataset_id_override_clears_builtin_hf_mapping(tmp_path) -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    config = TaskConfig(
        datasets=['gsm8k'],
        dataset_dir=str(tmp_path / 'evalscope'),
        dataset_hub=HubType.HUGGINGFACE,
        dataset_args={'gsm8k': {'dataset_id': 'custom/gsm8k'}},
    )
    benchmark = get_benchmark('gsm8k', config)

    assert benchmark.dataset_id == 'custom/gsm8k'
    assert benchmark.dataset_revision is None
    assert benchmark._benchmark_meta.hf_dataset_id is None
    assert benchmark._benchmark_meta.hf_dataset_revision is None


def test_hf_dataset_id_mapping_resolves_on_huggingface_hub(tmp_path) -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    config = TaskConfig(datasets=['gsm8k'], dataset_dir=str(tmp_path / 'evalscope'))
    benchmark = get_benchmark('gsm8k', config)
    benchmark._benchmark_meta.hf_dataset_id = 'openai/gsm8k'
    benchmark._benchmark_meta.hf_dataset_revision = 'refs/convert/parquet'

    assert benchmark.dataset_id == 'AI-ModelScope/gsm8k'
    assert benchmark.dataset_revision is None

    benchmark.dataset_hub = HubType.HUGGINGFACE

    assert benchmark.dataset_id == 'openai/gsm8k'
    assert benchmark.dataset_revision == 'refs/convert/parquet'


def test_hf_dataset_mapping_falls_back_without_mapping(tmp_path, monkeypatch) -> None:
    from evalscope.api.benchmark import benchmark as benchmark_module
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    config = TaskConfig(
        datasets=['gsm8k'], dataset_dir=str(tmp_path / 'evalscope'), dataset_hub=HubType.HUGGINGFACE,
    )
    benchmark = get_benchmark('gsm8k', config)
    benchmark._benchmark_meta.hf_dataset_id = None
    warnings = []
    monkeypatch.setattr(benchmark_module.logger, 'warning', lambda message: warnings.append(message))

    assert benchmark.dataset_id == 'AI-ModelScope/gsm8k'
    assert any('no hf_dataset_id mapping' in message for message in warnings)


def test_hf_dataset_mapping_ignored_for_local_and_modelscope_hubs(tmp_path) -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    config = TaskConfig(datasets=['gsm8k'], dataset_dir=str(tmp_path / 'evalscope'))
    benchmark = get_benchmark('gsm8k', config)
    benchmark._benchmark_meta.hf_dataset_id = 'openai/gsm8k'

    for hub in [HubType.MODELSCOPE, HubType.LOCAL]:
        benchmark.dataset_hub = hub
        assert benchmark.dataset_id == 'AI-ModelScope/gsm8k'


def test_local_path_override_clears_hf_mapping(tmp_path) -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    config = TaskConfig(
        datasets=['gsm8k'], dataset_dir=str(tmp_path / 'evalscope'), dataset_hub=HubType.HUGGINGFACE,
        dataset_args={'gsm8k': {'local_path': str(tmp_path / 'local-gsm8k')}},
    )
    benchmark = get_benchmark('gsm8k', config)

    assert benchmark.dataset_id == str(tmp_path / 'local-gsm8k')
    assert benchmark._benchmark_meta.hf_dataset_id is None
    assert benchmark._benchmark_meta.hf_dataset_revision is None


@pytest.mark.parametrize(
    ('benchmark_name', 'hf_dataset_id'),
    [('msvd', 'VLM2Vec/MSVD'), ('msr_vtt', 'VLM2Vec/MSR-VTT')],
)
def test_video_caption_adapters_use_shared_hf_dataset_mapping(
    benchmark_name: str,
    hf_dataset_id: str,
) -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    adapter = get_benchmark(
        benchmark_name,
        TaskConfig(datasets=[benchmark_name], dataset_hub=HubType.HUGGINGFACE),
    )

    assert adapter.dataset_id == hf_dataset_id
    assert adapter.source_dataset.data_id_or_path == hf_dataset_id
    if benchmark_name == 'msr_vtt':
        assert adapter.source_eval_split == 'test'
        assert adapter.source_subset == 'test_1k'


@pytest.mark.parametrize(
    ('dataset_id', 'expected_split', 'expected_subset'),
    [
        ('VLM2Vec/MSR-VTT', 'test', 'test_1k'),
        ('custom/MSR-VTT', 'validation', 'default'),
    ],
)
def test_msr_vtt_split_follows_effective_dataset_source(
    dataset_id: str,
    expected_split: str,
    expected_subset: str,
) -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    adapter = get_benchmark(
        'msr_vtt',
        TaskConfig(
            datasets=['msr_vtt'],
            dataset_hub=HubType.HUGGINGFACE,
            dataset_args={'msr_vtt': {'dataset_id': dataset_id}},
        ),
    )

    assert adapter.dataset_id == dataset_id
    assert adapter.source_eval_split == expected_split
    assert adapter.source_subset == expected_subset


def test_msr_vtt_explicit_official_hf_source_loads_official_split(monkeypatch) -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.benchmarks.msr_vtt.msr_vtt_adapter import MSRVTTAdapter
    from evalscope.config import TaskConfig

    captured = {}

    class CapturingDatasetHub:
        def load(self, split, subset):
            captured.update(split=split, subset=subset)
            return []

    adapter = get_benchmark(
        'msr_vtt',
        TaskConfig(
            datasets=['msr_vtt'],
            dataset_hub=HubType.HUGGINGFACE,
            dataset_args={'msr_vtt': {'dataset_id': 'VLM2Vec/MSR-VTT'}},
        ),
    )
    monkeypatch.setattr(MSRVTTAdapter, 'source_dataset', property(lambda self: CapturingDatasetHub()))

    assert adapter._load_records() == []
    assert captured == {'split': 'test', 'subset': 'test_1k'}


def test_aa_lcr_document_archive_uses_resolved_hub_source(tmp_path, monkeypatch) -> None:
    import zipfile

    from evalscope.api.registry import get_benchmark
    from evalscope.benchmarks.aa_lcr import aa_lcr_adapter as aa_lcr_module
    from evalscope.config import TaskConfig

    archive_path = tmp_path / 'documents.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('lcr/document.txt', 'content')
    captured = {}

    class CapturingDatasetHub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def download_file(self, file_path):
            captured['file_path'] = file_path
            return str(archive_path)

    monkeypatch.setattr(aa_lcr_module, 'DatasetHub', CapturingDatasetHub)
    monkeypatch.setattr(aa_lcr_module, 'DEFAULT_EVALSCOPE_CACHE_DIR', str(tmp_path / 'cache'))
    adapter = get_benchmark(
        'aa_lcr',
        TaskConfig(datasets=['aa_lcr'], dataset_hub=HubType.HUGGINGFACE),
    )

    extracted_dir = adapter._ensure_text_dir_downloaded()

    assert captured['data_id_or_path'] == 'ArtificialAnalysis/AA-LCR'
    assert captured['data_source'] == HubType.HUGGINGFACE
    assert captured['revision'] is None
    assert captured['file_path'] == 'extracted_text/AA-LCR_extracted-text.zip'
    assert (extracted_dir / 'document.txt').read_text(encoding='utf-8') == 'content'


def test_swe_bench_oracle_loader_uses_selected_hub_and_own_revision(monkeypatch) -> None:
    from evalscope.api.benchmark import DefaultDataAdapter
    from evalscope.api.dataset import Sample
    from evalscope.benchmarks.swe_bench import swe_bench_adapter as swe_module

    captured = {}

    class CapturingRemoteDataLoader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def load(self):
            return [Sample(input='problem', metadata={'instance_id': 'id', 'text': 'context'})]

    adapter = object.__new__(swe_module.SWEBenchVerifiedAdapter)
    adapter.inference_dataset_id = 'princeton-nlp/SWE-bench_oracle'
    adapter.inference_dataset_revision = 'oracle-revision'
    adapter._task_config = type('Config', (), {'dataset_hub': HubType.HUGGINGFACE})()
    monkeypatch.setattr(swe_module, 'RemoteDataLoader', CapturingRemoteDataLoader)
    monkeypatch.setattr(DefaultDataAdapter, 'load', lambda self: 'main-dataset')

    assert swe_module.SWEBenchVerifiedAdapter.load(adapter) == 'main-dataset'
    assert captured['data_id_or_path'] == 'princeton-nlp/SWE-bench_oracle'
    assert captured['data_source'] == HubType.HUGGINGFACE
    assert captured['version'] == 'oracle-revision'


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
