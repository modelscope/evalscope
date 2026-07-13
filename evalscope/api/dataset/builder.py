import copy
import random
from typing import Callable, Dict, Iterable, Optional, Union

from .dataset import DatasetDict, FieldSpec, MemoryDataset, Sample
from .hub import DatasetHub
from .loader import LocalDataLoader
from .utils import data_to_samples, record_to_sample_fn


def build_dataset_from_records(
    records: Iterable[dict],
    sample_fields: Union[FieldSpec, Callable],
    *,
    name: str,
    location: Optional[str],
    limit: Optional[Union[int, float]],
    repeats: int,
    shuffle: bool,
    seed: Optional[int],
    filter_func: Optional[Callable[[Sample], bool]] = None,
    auto_id: bool = True,
) -> MemoryDataset:
    """Build a MemoryDataset from raw records using the standard adapter mechanics.

    Note:
        ``repeats`` duplicates each *resulting sample* (not the raw record) ``repeats``
        times consecutively, then ``reindex`` groups them with ``group_size=repeats``.
        This matches the single-sample-per-record adapters. If ``sample_fields`` maps one
        record to multiple samples, the per-sample duplication order differs from record
        level duplication, which callers relying on k-metric grouping should be aware of.
    """
    record_list = list(records)
    if shuffle:
        random.Random(seed).shuffle(record_list)

    if limit is not None:
        if isinstance(limit, float):
            limit = int(len(record_list) * limit)
        elif isinstance(limit, int) and limit < 0:
            raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
        record_list = record_list[:limit]

    data_to_sample = record_to_sample_fn(sample_fields)
    samples = data_to_samples(data=record_list, data_to_sample=data_to_sample)
    if repeats > 1:
        samples = [copy.deepcopy(sample) for sample in samples for _ in range(repeats)]

    dataset = MemoryDataset(samples=samples, name=name, location=location, shuffled=shuffle)
    if filter_func is not None:
        dataset = dataset.filter(filter_func)
    if auto_id:
        dataset.reindex(group_size=repeats if repeats > 0 else 1)
    return dataset


def build_dataset_dict_from_record_map(
    record_map: Dict[str, Iterable[dict]],
    sample_fields: Union[FieldSpec, Callable],
    *,
    location: Optional[str],
    limit: Optional[Union[int, float]],
    repeats: int,
    shuffle: bool,
    seed: Optional[int],
    filter_func: Optional[Callable[[Sample], bool]] = None,
    auto_id: bool = True,
) -> DatasetDict:
    """Build a DatasetDict from a mapping of subset name to raw records."""
    datasets = {}
    for subset, records in record_map.items():
        datasets[subset] = build_dataset_from_records(
            records=records,
            sample_fields=sample_fields,
            name=subset,
            location=location,
            limit=limit,
            repeats=repeats,
            shuffle=shuffle,
            seed=seed,
            filter_func=filter_func,
            auto_id=auto_id,
        )
    return DatasetDict(datasets)


def resolve_snapshot_or_local_path(adapter, allow_file_pattern=None) -> str:
    """Resolve an adapter dataset_id as a local path or downloaded snapshot root."""
    return DatasetHub(
        data_id_or_path=adapter.dataset_id,
        data_source=adapter.dataset_hub,
        force_redownload=adapter.force_redownload,
        cache_dir=adapter.dataset_dir,
    ).download_snapshot(allow_file_pattern=allow_file_pattern)


def load_local_file_dataset(
    adapter,
    dataset_path: str,
    subset: str,
    split: str,
    sample_fields: Union[FieldSpec, Callable],
    limit: Optional[Union[int, float]],
    repeats: int,
    shuffle: bool,
) -> MemoryDataset:
    """Load a local JSONL/CSV/TSV file or directory with the standard LocalDataLoader."""
    return LocalDataLoader(
        data_id_or_path=dataset_path,
        split=split,
        subset=subset,
        sample_fields=sample_fields,
        limit=limit,
        repeats=repeats,
        shuffle=shuffle,
    ).load()
