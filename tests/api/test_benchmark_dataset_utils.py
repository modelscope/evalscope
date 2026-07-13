from typing import Any, ClassVar, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.benchmark.adapters.dataset_utils import (
    build_dataset_dict_from_record_map,
    build_dataset_from_records,
    load_local_file_dataset,
    resolve_snapshot_or_local_path,
)
from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.config import TaskConfig


def record_to_sample(record: Dict[str, Any]) -> Sample:
    return Sample(input=str(record['value']), metadata={'value': record['value']})


def test_build_dataset_from_records_applies_limit_repeats_and_reindex() -> None:
    dataset = build_dataset_from_records(
        records=[{'value': 1}, {'value': 2}, {'value': 3}],
        sample_fields=record_to_sample,
        name='subset',
        location='source',
        limit=2,
        repeats=2,
        shuffle=False,
        seed=None,
    )

    assert isinstance(dataset, MemoryDataset)
    assert [sample.input for sample in dataset] == ['1', '1', '2', '2']
    assert [sample.id for sample in dataset] == [0, 1, 2, 3]
    assert [sample.group_id for sample in dataset] == [0, 0, 1, 1]
    assert dataset.name == 'subset'
    assert dataset.location == 'source'


def test_build_dataset_from_records_filter_runs_before_reindex() -> None:
    dataset = build_dataset_from_records(
        records=[{'value': 1}, {'value': 2}, {'value': 3}],
        sample_fields=record_to_sample,
        name='subset',
        location=None,
        limit=None,
        repeats=1,
        shuffle=False,
        seed=None,
        filter_func=lambda sample: sample.metadata['value'] != 2,
    )

    assert [sample.input for sample in dataset] == ['1', '3']
    assert [sample.id for sample in dataset] == [0, 1]


def test_build_dataset_dict_from_record_map_uses_stable_shuffle_seed() -> None:
    record_map = {'a': [{'value': idx} for idx in range(6)]}

    first = build_dataset_dict_from_record_map(
        record_map,
        sample_fields=record_to_sample,
        location='source',
        limit=3,
        repeats=1,
        shuffle=True,
        seed=7,
    )
    second = build_dataset_dict_from_record_map(
        record_map,
        sample_fields=record_to_sample,
        location='source',
        limit=3,
        repeats=1,
        shuffle=True,
        seed=7,
    )

    assert [sample.input for sample in first['a']] == [sample.input for sample in second['a']]
    assert [sample.input for sample in first['a']] != ['0', '1', '2']


class DummyAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return record_to_sample(record)


def make_adapter(dataset_id: str) -> DummyAdapter:
    return DummyAdapter(
        benchmark_meta=BenchmarkMeta(name='dummy', dataset_id=dataset_id, eval_split='test'),
        task_config=TaskConfig(datasets=['dummy']),
    )


def test_resolve_snapshot_or_local_path_uses_adapter_hub(tmp_path) -> None:
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()
    adapter = make_adapter(str(dataset_dir))

    assert resolve_snapshot_or_local_path(adapter) == str(dataset_dir.resolve())


def test_load_local_file_dataset_passes_loader_args(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    class CapturingLocalDataLoader:
        latest_kwargs: ClassVar[Dict[str, Any]] = {}

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def load(self) -> MemoryDataset:
            return MemoryDataset([Sample(input='ok')], name='loaded')

    monkeypatch.setattr(
        'evalscope.api.benchmark.adapters.dataset_utils.LocalDataLoader',
        CapturingLocalDataLoader,
    )

    adapter = make_adapter('/tmp/data')
    dataset = load_local_file_dataset(
        adapter=adapter,
        dataset_path='/tmp/data/file.jsonl',
        subset='test',
        split='validation',
        sample_fields=record_to_sample,
        limit=5,
        repeats=2,
        shuffle=True,
    )

    assert dataset.name == 'loaded'
    assert captured == {
        'data_id_or_path': '/tmp/data/file.jsonl',
        'split': 'validation',
        'subset': 'test',
        'sample_fields': record_to_sample,
        'limit': 5,
        'repeats': 2,
        'shuffle': True,
    }
