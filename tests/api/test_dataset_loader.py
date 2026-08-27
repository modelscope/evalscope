"""Tests for dataset loader cache isolation, media handling, and limits."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetInfo, Features, Value, Video
from datasets.table import InMemoryTable

from evalscope.api.dataset import Sample
from evalscope.api.dataset.hub import DatasetHub
from evalscope.api.dataset.loader import DictDataLoader, LocalDataLoader, RemoteDataLoader
from evalscope.constants import HubType


def _sample_from_record(record: Dict[str, Any]) -> Sample:
    return Sample(input=record['text'], metadata={'video': record.get('video')})


def _install_fake_remote_dataset(monkeypatch: Any) -> None:
    def fake_load(self: DatasetHub, split: str, subset: str = 'default', **kwargs: Any) -> HFDataset:
        return HFDataset.from_dict({'text': [f'{subset}-{split}']})

    monkeypatch.setattr(DatasetHub, 'load', fake_load)


def _cache_directories(dataset_dir: Path) -> List[Path]:
    cache_root = dataset_dir / 'datasets'
    return sorted(path for path in cache_root.iterdir() if path.is_dir())


def _load_remote_cache(
    dataset_dir: Path,
    *,
    data_source: str,
    split: str,
    subset: str,
    loader_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    RemoteDataLoader(
        data_id_or_path='dummy/cache-dataset',
        split=split,
        subset=subset,
        sample_fields=_sample_from_record,
        data_source=data_source,
        dataset_dir=str(dataset_dir),
        **(loader_kwargs or {}),
    ).load()


def test_remote_cache_directory_separates_data_sources(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fake_remote_dataset(monkeypatch)

    _load_remote_cache(tmp_path, data_source=HubType.MODELSCOPE, split='train', subset='default')
    _load_remote_cache(tmp_path, data_source=HubType.HUGGINGFACE, split='train', subset='default')

    assert len(_cache_directories(tmp_path)) == 2


def test_remote_cache_directory_preserves_split_subset_boundaries(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fake_remote_dataset(monkeypatch)

    _load_remote_cache(tmp_path, data_source=HubType.MODELSCOPE, split='a', subset='bc')
    _load_remote_cache(tmp_path, data_source=HubType.MODELSCOPE, split='ab', subset='c')

    assert len(_cache_directories(tmp_path)) == 2


def test_remote_cache_directory_is_stable_across_kwargs_order(monkeypatch: Any, tmp_path: Path) -> None:
    _install_fake_remote_dataset(monkeypatch)

    _load_remote_cache(
        tmp_path,
        data_source=HubType.MODELSCOPE,
        split='train',
        subset='default',
        loader_kwargs={'b': 2, 'a': 1},
    )
    _load_remote_cache(
        tmp_path,
        data_source=HubType.MODELSCOPE,
        split='train',
        subset='default',
        loader_kwargs={'a': 1, 'b': 2},
    )

    assert len(_cache_directories(tmp_path)) == 1


def test_remote_loader_treats_existing_path_as_effective_local_source(monkeypatch: Any, tmp_path: Path) -> None:
    local_path = tmp_path / 'local-dataset'
    local_path.mkdir()
    dataset_dir = tmp_path / 'cache'
    captured: Dict[str, Any] = {}

    def fake_load(self: DatasetHub, split: str, subset: str = 'default', **kwargs: Any) -> HFDataset:
        captured['data_source'] = self.data_source
        return HFDataset.from_dict({'text': ['local']})

    monkeypatch.setattr(DatasetHub, 'load', fake_load)

    dataset = RemoteDataLoader(
        data_id_or_path=str(local_path),
        split='train',
        sample_fields=_sample_from_record,
        data_source=HubType.MODELSCOPE,
        dataset_dir=str(dataset_dir),
    ).load()

    assert len(dataset) == 1
    assert captured['data_source'] == HubType.LOCAL
    assert not (dataset_dir / 'datasets').exists()


def test_remote_loader_does_not_decode_video(monkeypatch: Any, tmp_path: Path) -> None:
    features = Features({'video': Video(decode=True), 'text': Value('string')})
    row = {'video': {'bytes': b'dummy', 'path': None}, 'text': 'clip'}
    table = pa.Table.from_pylist([row], schema=features.arrow_schema)
    remote_dataset = HFDataset(InMemoryTable(table), info=DatasetInfo(features=features))

    def fake_load(self: DatasetHub, split: str, subset: str = 'default', **kwargs: Any) -> HFDataset:
        return remote_dataset

    monkeypatch.setattr(DatasetHub, 'load', fake_load)

    dataset = RemoteDataLoader(
        data_id_or_path='dummy/video-dataset',
        split='train',
        sample_fields=_sample_from_record,
        data_source=HubType.LOCAL,
        dataset_dir=str(tmp_path),
    ).load()

    assert dataset[0].metadata['video'] == {'bytes': b'dummy', 'path': None}


def _make_direct_loader(
    loader_kind: str,
    tmp_path: Path,
    monkeypatch: Any,
    limit: Optional[float | int],
) -> RemoteDataLoader | LocalDataLoader | DictDataLoader:
    records = [{'text': f'row-{index}'} for index in range(4)]
    if loader_kind == 'remote':
        monkeypatch.setattr(DatasetHub, 'load', lambda self, split, subset='default', **kwargs: HFDataset.from_list(records))
        return RemoteDataLoader(
            data_id_or_path='dummy/limit-dataset',
            split='train',
            sample_fields=_sample_from_record,
            data_source=HubType.LOCAL,
            dataset_dir=str(tmp_path),
            limit=limit,
        )
    if loader_kind == 'local':
        local_path = tmp_path / 'records.jsonl'
        local_path.write_text('\n'.join(json.dumps(record) for record in records), encoding='utf-8')
        return LocalDataLoader(
            data_id_or_path=str(local_path),
            split='train',
            sample_fields=_sample_from_record,
            limit=limit,
        )
    return DictDataLoader(dict_list=records, sample_fields=_sample_from_record, limit=limit)


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
@pytest.mark.parametrize('limit', [2.5, -1, -0.1, math.nan, math.inf])
def test_direct_loaders_reject_invalid_limits(
    loader_kind: str,
    limit: float | int,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match='Limit must'):
        _make_direct_loader(loader_kind, tmp_path, monkeypatch, limit).load()


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
@pytest.mark.parametrize(
    'limit, expected',
    [
        (None, 4),
        (0, 4),
        (0.5, 2),
        (1.0, 4),
        (2, 2),
    ],
)
def test_direct_loaders_preserve_valid_limit_semantics(
    loader_kind: str,
    limit: Optional[float | int],
    expected: int,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    loader = _make_direct_loader(loader_kind, tmp_path, monkeypatch, limit)

    assert len(loader.load()) == expected
    assert len(loader.load()) == expected
    assert loader.limit == limit
