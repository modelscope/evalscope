"""The sample limit is spent on samples that pass the filter, not on raw records the filter later drops."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pytest
from datasets import Dataset as HFDataset

from evalscope.api.dataset import Sample, build_dataset_from_records
from evalscope.api.dataset.hub import DatasetHub
from evalscope.api.dataset.loader import DictDataLoader, LocalDataLoader, RemoteDataLoader
from evalscope.constants import HubType

RECORDS = [{'text': f'row-{index}'} for index in range(10)]


def _row(sample: Sample) -> int:
    assert isinstance(sample.input, str)
    return int(sample.input.split('-')[1])


def _to_sample(record: Dict[str, Any]) -> Sample:
    return Sample(input=record['text'])


def _late_rows_only(sample: Sample) -> bool:
    """Rejects the first six rows, the way a date or needle-count filter rejects the head of a sorted dataset."""
    return _row(sample) >= 6


def _make_loader(
    loader_kind: str,
    tmp_path: Path,
    monkeypatch: Any,
    *,
    limit: Optional[Union[int, float]],
    repeats: int = 1,
    sample_fields: Callable[[Dict[str, Any]], Union[Sample, List[Sample]]] = _to_sample,
    filter_func: Optional[Callable[[Sample], bool]] = None,
) -> Union[RemoteDataLoader, LocalDataLoader, DictDataLoader]:
    kwargs: Dict[str, Any] = dict(
        sample_fields=sample_fields, limit=limit, repeats=repeats, filter_func=filter_func, auto_id=True
    )
    if loader_kind == 'remote':
        monkeypatch.setattr(
            DatasetHub, 'load', lambda self, split, subset='default', **kw: HFDataset.from_list(RECORDS)
        )
        return RemoteDataLoader(
            data_id_or_path='dummy/limit-filter-dataset',
            split='train',
            data_source=HubType.LOCAL,
            dataset_dir=str(tmp_path),
            **kwargs,
        )
    if loader_kind == 'local':
        path = tmp_path / 'records.jsonl'
        path.write_text('\n'.join(json.dumps(record) for record in RECORDS), encoding='utf-8')
        return LocalDataLoader(data_id_or_path=str(path), split='train', **kwargs)
    return DictDataLoader(dict_list=RECORDS, **kwargs)


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
def test_limit_counts_only_samples_that_pass_the_filter(loader_kind: str, tmp_path: Path, monkeypatch: Any) -> None:
    loader = _make_loader(loader_kind, tmp_path, monkeypatch, limit=3, filter_func=_late_rows_only)

    dataset = loader.load()

    assert [_row(sample) for sample in dataset] == [6, 7, 8]
    assert [sample.id for sample in dataset] == [0, 1, 2]


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
def test_fractional_limit_is_resolved_against_records_and_filled_by_passing_samples(
    loader_kind: str, tmp_path: Path, monkeypatch: Any
) -> None:
    # 0.5 of 10 records is 5 samples, but only rows 6-9 pass the filter.
    loader = _make_loader(loader_kind, tmp_path, monkeypatch, limit=0.5, filter_func=_late_rows_only)

    assert [_row(sample) for sample in loader.load()] == [6, 7, 8, 9]


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
def test_repeats_group_the_samples_that_pass_the_filter(loader_kind: str, tmp_path: Path, monkeypatch: Any) -> None:
    loader = _make_loader(loader_kind, tmp_path, monkeypatch, limit=2, repeats=2, filter_func=_late_rows_only)

    dataset = loader.load()

    assert [_row(sample) for sample in dataset] == [6, 6, 7, 7]
    assert [sample.group_id for sample in dataset] == [0, 0, 1, 1]


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
def test_limit_zero_still_selects_nothing_with_a_filter(loader_kind: str, tmp_path: Path, monkeypatch: Any) -> None:
    loader = _make_loader(loader_kind, tmp_path, monkeypatch, limit=0, filter_func=_late_rows_only)

    assert len(loader.load()) == 0


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
def test_limit_counts_only_records_that_produce_a_sample(loader_kind: str, tmp_path: Path, monkeypatch: Any) -> None:
    def skip_first_six(record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        sample = _to_sample(record)
        return sample if _row(sample) >= 6 else []

    loader = _make_loader(loader_kind, tmp_path, monkeypatch, limit=3, sample_fields=skip_first_six)

    assert [_row(sample) for sample in loader.load()] == [6, 7, 8]


@pytest.mark.parametrize('loader_kind', ['remote', 'local', 'dict'])
def test_records_past_the_limit_are_not_converted(loader_kind: str, tmp_path: Path, monkeypatch: Any) -> None:
    converted: List[str] = []

    def record_calls(record: Dict[str, Any]) -> Sample:
        converted.append(record['text'])
        return _to_sample(record)

    loader = _make_loader(loader_kind, tmp_path, monkeypatch, limit=2, sample_fields=record_calls)

    assert len(loader.load()) == 2
    assert converted == ['row-0', 'row-1']


def test_builder_limit_counts_only_samples_that_pass_the_filter() -> None:
    dataset = build_dataset_from_records(
        records=RECORDS,
        sample_fields=_to_sample,
        name='subset',
        location='source',
        limit=3,
        repeats=1,
        shuffle=False,
        seed=None,
        filter_func=_late_rows_only,
    )

    assert [_row(sample) for sample in dataset] == [6, 7, 8]
    assert [sample.id for sample in dataset] == [0, 1, 2]
