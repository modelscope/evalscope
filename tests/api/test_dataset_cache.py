"""Dataset cache publication and temporary ModelScope parsing lifecycle."""

import errno
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import datasets
import pytest
from datasets import Dataset as HFDataset
from datasets.packaged_modules.json.json import Json

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample, hub, loader
from evalscope.api.dataset.hub import DatasetHub
from evalscope.api.dataset.loader import RemoteDataLoader
from evalscope.config import TaskConfig


class TextAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        return Sample(input=record['text'])


def _make_loader(root: Path, **kwargs: Any) -> RemoteDataLoader:
    return RemoteDataLoader(
        data_id_or_path='owner/data',
        split='test',
        sample_fields=lambda record: Sample(input=record['text']),
        dataset_dir=str(root),
        **kwargs,
    )


def _only_cache(root: Path) -> Path:
    directories = [path for path in (root / 'datasets').iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


def _install_records(monkeypatch: pytest.MonkeyPatch, text: str = 'old') -> None:
    monkeypatch.setattr(DatasetHub, 'load', lambda *args, **kwargs: HFDataset.from_dict({'text': [text]}))


def test_benchmark_shares_temporary_parsing_and_keeps_only_evalscope_arrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    for split in ['test', 'train']:
        (snapshot / f'{split}.jsonl').write_text(json.dumps({'text': split}) + '\n', encoding='utf-8')
    (snapshot / 'README.md').write_text(
        '---\nconfigs:\n- config_name: default\n  data_files:\n'
        '  - split: test\n    path: test.jsonl\n  - split: train\n    path: train.jsonl\n---\n',
        encoding='utf-8',
    )
    persistent_hf = tmp_path / 'hf'
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', persistent_hf)
    monkeypatch.setattr(datasets.config, 'DOWNLOADED_DATASETS_PATH', persistent_hf / 'downloads')
    monkeypatch.setattr(hub, '_try_modelscope_cached_snapshot', lambda *args, **kwargs: str(snapshot))
    original_load = datasets.load_dataset
    original_generate = Json._generate_tables
    parsing_dirs = []
    download_dirs = []
    generations = []

    def count_generation(self: Json, *args: Any, **kwargs: Any) -> Any:
        # JSON also reads a small batch to infer features before generating the splits.
        if kwargs.get('allow_full_read', True):
            generations.append(True)
        yield from original_generate(self, *args, **kwargs)

    def capture_load(**kwargs: Any) -> HFDataset:
        parsing_dirs.append(kwargs['cache_dir'])
        download_dirs.append(kwargs['download_config'].cache_dir)
        return original_load(**kwargs)

    monkeypatch.setattr(datasets, 'load_dataset', capture_load)
    monkeypatch.setattr(Json, '_generate_tables', count_generation)
    cache_root = tmp_path / 'evalscope'
    adapter = TextAdapter(
        benchmark_meta=BenchmarkMeta(
            name='cache_smoke', dataset_id='owner/data', subset_list=['default'],
            eval_split='test', train_split='train', few_shot_num=1,
        ),
        task_config=TaskConfig(datasets=['cache_smoke'], dataset_dir=str(cache_root)),
    )

    test, fewshot = adapter.load_from_remote()

    assert test['default'][0].input == 'test'
    assert fewshot['default'][0].input == 'train'
    assert len(parsing_dirs) == 2 and parsing_dirs[0] == parsing_dirs[1]
    assert len(generations) == 2
    assert not Path(parsing_dirs[0]).exists()
    assert download_dirs == [str(persistent_hf / 'downloads')] * 2
    assert not list(persistent_hf.rglob('*.arrow'))
    assert len(list(cache_root.rglob('*.arrow'))) == 2

    def forbid_load(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('A processed cache hit must skip the hub and temporary parsing')

    monkeypatch.setattr(DatasetHub, 'load', forbid_load)
    monkeypatch.setattr(loader._DatasetLoadingSession, 'snapshot_cache_dir', forbid_load)
    repeated_test, repeated_fewshot = adapter.load_from_remote()
    assert repeated_test['default'][0].input == 'test'
    assert repeated_fewshot['default'][0].input == 'train'


@pytest.mark.parametrize('failure', ['load', 'save', 'validate', 'publish'])
def test_failed_force_refresh_keeps_the_previous_cache(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_records(monkeypatch)
    assert _make_loader(tmp_path).load()[0].input == 'old'
    cache = _only_cache(tmp_path)
    _install_records(monkeypatch, 'new')

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError('Simulated refresh failure')

    if failure == 'load':
        monkeypatch.setattr(DatasetHub, 'load', fail)
    elif failure == 'save':
        monkeypatch.setattr(HFDataset, 'save_to_disk', fail)
    elif failure == 'validate':
        original_load = datasets.load_from_disk

        def fail_validation(path: str, **kwargs: Any) -> HFDataset:
            if str(path).endswith('.incomplete'):
                fail()
            return original_load(path, **kwargs)

        monkeypatch.setattr(datasets, 'load_from_disk', fail_validation)
    else:
        original_replace = os.replace

        def fail_publication(source: str, destination: str) -> None:
            if str(source).endswith('.incomplete'):
                fail()
            original_replace(source, destination)

        monkeypatch.setattr(loader.os, 'replace', fail_publication)

    with pytest.raises(OSError, match='Simulated refresh failure'):
        _make_loader(tmp_path, force_redownload=True).load()

    with datasets.load_from_disk(cache) as previous:
        assert previous['text'] == ['old']
    assert _make_loader(tmp_path).load()[0].input == 'old'
    assert _only_cache(tmp_path) == cache


def test_failed_initial_save_does_not_publish_an_incomplete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_records(monkeypatch)
    original_save = HFDataset.save_to_disk

    def interrupted_save(self: HFDataset, path: str) -> None:
        Path(path).mkdir()
        raise OSError('Interrupted write')

    monkeypatch.setattr(HFDataset, 'save_to_disk', interrupted_save)
    with pytest.raises(OSError, match='Interrupted write'):
        _make_loader(tmp_path).load()
    assert not [path for path in (tmp_path / 'datasets').iterdir() if path.is_dir()]

    monkeypatch.setattr(HFDataset, 'save_to_disk', original_save)
    assert _make_loader(tmp_path).load()[0].input == 'old'


def test_failed_publication_and_rollback_do_not_delete_the_old_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_records(monkeypatch)
    _make_loader(tmp_path).load()
    _install_records(monkeypatch, 'new')
    original_replace = os.replace

    def fail_publish_and_restore(source: str, destination: str) -> None:
        if str(source).endswith(('.incomplete', '.previous')):
            raise OSError('Filesystem cannot publish or restore')
        original_replace(source, destination)

    monkeypatch.setattr(loader.os, 'replace', fail_publish_and_restore)
    with pytest.raises(OSError, match='cannot publish or restore'):
        _make_loader(tmp_path, force_redownload=True).load()
    backup = _only_cache(tmp_path)
    assert backup.name.endswith('.previous')
    with datasets.load_from_disk(backup) as previous:
        assert previous['text'] == ['old']
    monkeypatch.setattr(loader.os, 'replace', original_replace)

    def forbid_hub(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('A recovered previous cache must not access the hub')

    monkeypatch.setattr(DatasetHub, 'load', forbid_hub)
    assert _make_loader(tmp_path).load()[0].input == 'old'
    assert not backup.exists()


@pytest.mark.parametrize('phase', ['prepare', 'backup', 'publish'])
def test_process_exit_during_publication_recovers_without_the_hub(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_records(monkeypatch)
    _make_loader(tmp_path).load()
    canonical = _only_cache(tmp_path)
    worktree = Path(__file__).resolve().parents[2]
    child = '''
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch
sys.path.insert(0, sys.argv[2])
import datasets
from evalscope.api.dataset import Sample, loader
from evalscope.api.dataset.hub import DatasetHub
original_replace = os.replace
original_save = datasets.Dataset.save_to_disk
def exit_after_prepare(self: datasets.Dataset, *args: Any, **kwargs: Any) -> Any:
    result = original_save(self, *args, **kwargs)
    if sys.argv[3] == 'prepare':
        os._exit(86)
    return result
def exit_at_publication(source: str, destination: str) -> None:
    original_replace(source, destination)
    name = Path(destination).name
    is_backup = '-previous-' in name or name.endswith('.previous')
    is_canonical = str(destination) == sys.argv[4]
    if (sys.argv[3] == 'backup' and is_backup) or (sys.argv[3] == 'publish' and is_canonical):
        os._exit(86)
loader.os.replace = exit_at_publication
datasets.Dataset.save_to_disk = exit_after_prepare
with patch.object(DatasetHub, 'load', lambda *args, **kwargs: datasets.Dataset.from_dict({'text': ['new']})):
    loader.RemoteDataLoader(
        data_id_or_path='owner/data', split='test', dataset_dir=sys.argv[1], force_redownload=True,
        sample_fields=lambda record: Sample(input=record['text']),
    ).load()
'''
    process = subprocess.run(
        [sys.executable, '-c', child, str(tmp_path), str(worktree), phase, str(canonical)],
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 86, process.stderr

    def forbid_hub(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Restart must recover the previous cache or use the committed cache without the hub')

    monkeypatch.setattr(DatasetHub, 'load', forbid_hub)
    assert _make_loader(tmp_path).load()[0].input == ('new' if phase == 'publish' else 'old')
    assert _only_cache(tmp_path) == canonical


def test_recovery_keeps_backup_if_published_cache_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_records(monkeypatch)
    _make_loader(tmp_path).load()
    canonical = _only_cache(tmp_path)
    backup = Path(f'{canonical}.previous')
    os.replace(canonical, backup)
    HFDataset.from_dict({'text': ['new']}).save_to_disk(canonical)
    (canonical / 'state.json').write_text('invalid json', encoding='utf-8')

    with pytest.raises(json.JSONDecodeError):
        _make_loader(tmp_path).load()
    with datasets.load_from_disk(backup) as previous:
        assert previous['text'] == ['old']


def test_missing_arrow_rebuilds_but_corrupt_metadata_still_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_records(monkeypatch)
    _make_loader(tmp_path).load()
    cache = _only_cache(tmp_path)
    next(cache.glob('*.arrow')).unlink()
    _install_records(monkeypatch, 'recovered')
    assert _make_loader(tmp_path).load()[0].input == 'recovered'

    (cache / 'state.json').write_text('invalid json', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        _make_loader(tmp_path).load()


def test_concurrent_cache_misses_build_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    start = Barrier(2)
    calls = []

    def remote_load(*args: Any, **kwargs: Any) -> HFDataset:
        calls.append(True)
        return HFDataset.from_dict({'text': ['shared']})

    def load() -> str:
        start.wait(timeout=10)
        return _make_loader(tmp_path).load()[0].input

    monkeypatch.setattr(DatasetHub, 'load', remote_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(load)
        second = pool.submit(load)
        assert [first.result(timeout=20), second.result(timeout=20)] == ['shared', 'shared']
    assert len(calls) == 1
    _only_cache(tmp_path)


def test_explicit_hf_cache_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'test.jsonl').write_text('{"text": "explicit"}\n', encoding='utf-8')
    monkeypatch.setattr(hub, '_try_modelscope_cached_snapshot', lambda *args, **kwargs: str(snapshot))
    explicit = tmp_path / 'explicit_hf'
    assert _make_loader(tmp_path / 'evalscope', cache_dir=str(explicit)).load()[0].input == 'explicit'
    assert list(explicit.rglob('*.arrow'))
    assert not list((tmp_path / 'evalscope').rglob('.hf-staging-*'))


def test_session_cleans_up_on_error_and_restores_nested_context(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='conversion failed'):
        with loader.dataset_loading_session() as outer:
            temporary = Path(outer.snapshot_cache_dir(str(tmp_path)))
            with loader.dataset_loading_session() as inner:
                assert inner is outer
            assert temporary.exists()
            raise RuntimeError('conversion failed')
    assert not temporary.exists()
    with loader.dataset_loading_session() as fresh:
        assert fresh is not outer


@pytest.mark.parametrize('error_number', [errno.EACCES, errno.EROFS])
@pytest.mark.parametrize('previous_only', [False, True])
def test_read_only_processed_cache_still_loads(
    error_number: int, previous_only: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_records(monkeypatch)
    _make_loader(tmp_path).load()
    if previous_only:
        canonical = _only_cache(tmp_path)
        os.replace(canonical, f'{canonical}.previous')

    def cannot_lock(*args: Any, **kwargs: Any) -> Any:
        raise OSError(error_number, 'Read-only cache')

    def forbid_hub(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Read-only cache hits must load without the hub')

    monkeypatch.setattr(loader.FileLock, 'acquire', cannot_lock)
    monkeypatch.setattr(loader.os, 'access', lambda *args, **kwargs: False)
    monkeypatch.setattr(DatasetHub, 'load', forbid_hub)
    assert _make_loader(tmp_path).load()[0].input == 'old'
    with pytest.raises(OSError, match='Read-only cache'):
        _make_loader(tmp_path, force_redownload=True).load()


def test_conversion_error_cleans_temporary_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'test.jsonl').write_text('{"text": "failure"}\n', encoding='utf-8')
    monkeypatch.setattr(hub, '_try_modelscope_cached_snapshot', lambda *args, **kwargs: str(snapshot))

    def fail_conversion(record: dict[str, Any]) -> Sample:
        raise RuntimeError('Conversion failed')

    data_loader = _make_loader(tmp_path / 'evalscope')
    data_loader.sample_fields = fail_conversion
    with pytest.raises(RuntimeError, match='Conversion failed'):
        data_loader.load()
    assert not list((tmp_path / 'evalscope').rglob('.hf-staging-*'))
    with datasets.load_from_disk(_only_cache(tmp_path / 'evalscope')) as saved:
        assert saved['text'] == ['failure']


def test_typed_media_is_embedded_before_temporary_resources_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = []

    def remote_load(*args: Any, **kwargs: Any) -> HFDataset:
        resource = Path(kwargs['_snapshot_cache_dir']()) / 'asset.png'
        resource.write_bytes(b'image payload')
        resources.append(resource)
        return HFDataset.from_dict(
            {'text': ['image'], 'image': [str(resource)]},
            features=datasets.Features({'text': datasets.Value('string'), 'image': datasets.Image(decode=False)}),
        )

    monkeypatch.setattr(DatasetHub, 'load', remote_load)
    data_loader = _make_loader(tmp_path)
    data_loader.sample_fields = lambda record: Sample(input=record['text'], metadata={'image': record['image']})

    result = data_loader.load()

    assert not resources[0].exists()
    assert result[0].metadata['image']['bytes'] == b'image payload'
    assert data_loader.load()[0].metadata['image']['bytes'] == b'image payload'
    assert len(list(tmp_path.rglob('*.arrow'))) == 1
    assert not list(tmp_path.rglob('.hf-staging-*'))


def test_download_options_and_resource_cache_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'test.jsonl').write_text('{"text": "resource"}\n', encoding='utf-8')
    monkeypatch.setattr(hub, '_try_modelscope_cached_snapshot', lambda *args, **kwargs: str(snapshot))
    download_config = datasets.DownloadConfig(cache_dir=str(tmp_path / 'resources'), max_retries=7)
    original_load = datasets.load_dataset

    def capture_load(**kwargs: Any) -> HFDataset:
        assert kwargs['download_config'] is not download_config
        assert kwargs['download_config'].cache_dir == download_config.cache_dir
        assert kwargs['download_config'].max_retries == 7
        assert kwargs['cache_dir'] != download_config.cache_dir
        return original_load(**kwargs)

    monkeypatch.setattr(datasets, 'load_dataset', capture_load)
    assert _make_loader(tmp_path / 'evalscope', download_config=download_config).load()[0].input == 'resource'
    assert download_config.cache_dir == str(tmp_path / 'resources')
    assert download_config.max_retries == 7


def test_script_builder_does_not_use_temporary_arrow_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class ScriptBuilder:
        config = SimpleNamespace(data_files={'test': [str(tmp_path / 'test.jsonl')]})

    monkeypatch.setattr(datasets, 'load_dataset_builder', lambda **kwargs: ScriptBuilder())

    def forbid_temporary_dir() -> str:
        pytest.fail('Script resources may depend on the persistent builder directory')

    result = hub._prepare_cached_dataset_kwargs({'path': str(tmp_path), 'split': 'test'}, forbid_temporary_dir)
    assert 'cache_dir' not in result
