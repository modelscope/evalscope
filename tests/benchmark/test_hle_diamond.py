from typing import Any
from unittest import mock

import pytest

from evalscope.api.dataset import DatasetDict, DictDataLoader
from evalscope.api.messages import ContentImage
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.hle_diamond.hle_diamond_adapter import HLEDiamondAdapter
from evalscope.config import TaskConfig


def _record(index: int, partition: str = 'reasoning', image: str = '') -> dict[str, Any]:
    return {
        'id': f'synthetic-{index}',
        'question': f'Synthetic question {index}',
        'answer': 'A',
        'answer_type': 'multipleChoice',
        'author_name': 'synthetic',
        'rationale': 'synthetic rationale',
        'raw_subject': 'synthetic subject',
        'category': 'Math',
        'image': image,
        'partition': partition,
    }


def _adapter(include_multi_modal: bool = True) -> HLEDiamondAdapter:
    return get_benchmark(
        'hle_diamond',
        TaskConfig(
            model='mock_llm',
            datasets=['hle_diamond'],
            dataset_args={'hle_diamond': {'extra_params': {'include_multi_modal': include_multi_modal}}},
        ),
    )


def _load_dataset(adapter: HLEDiamondAdapter, records: list[dict[str, Any]]) -> DatasetDict:

    class RecordsLoader(DictDataLoader):

        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop('data_id_or_path', None)
            kwargs.pop('split', None)
            super().__init__(records, **kwargs)

    with mock.patch('evalscope.api.benchmark.adapters.default_data_adapter.RemoteDataLoader', RecordsLoader):
        return adapter.load_dataset()


def test_registration_preserves_original_hle_grouping() -> None:
    diamond = _adapter()
    original = get_benchmark('hle')

    assert diamond.dataset_id == 'cais/hle-diamond'
    assert diamond.benchmark_meta.hf_dataset_id == 'cais/hle-diamond'
    assert diamond.subset_list == ['reasoning', 'knowledge']
    assert diamond.benchmark_meta.evaluation_version == 'v1.0'
    assert diamond._suppress_doc_sample_example is True

    original_sample = original.record_to_sample(_record(0))
    assert original_sample.subset_key == 'Math'
    assert 'partition' not in original_sample.metadata


def test_mocked_schema_loads_500_samples_per_partition() -> None:
    records = [_record(index, 'reasoning') for index in range(500)]
    records += [_record(index + 500, 'knowledge') for index in range(500)]

    dataset = _load_dataset(_adapter(), records)

    assert set(dataset.keys()) == {'reasoning', 'knowledge'}
    assert len(dataset['reasoning']) == 500
    assert len(dataset['knowledge']) == 500
    for partition in dataset.keys():
        assert all(sample.subset_key == partition for sample in dataset[partition])
        assert all(sample.metadata['partition'] == partition for sample in dataset[partition])
        assert all(sample.metadata['category'] == 'Math' for sample in dataset[partition])


def test_multimodal_conversion_and_text_only_filtering() -> None:
    records = [_record(0, image='data:image/png;base64,c3ludGhldGlj'), _record(1)]

    dataset = _load_dataset(_adapter(), records)
    image_sample = dataset['reasoning'][0]
    assert isinstance(image_sample.input[1].content[1], ContentImage)
    assert image_sample.metadata['has_image'] is True

    text_only = _load_dataset(_adapter(include_multi_modal=False), records)
    assert len(text_only['reasoning']) == 1
    assert text_only['reasoning'][0].metadata['uid'] == 'synthetic-1'


def test_invalid_partition_fails_instead_of_dropping_the_sample() -> None:
    with pytest.raises(ValueError, match='Unsupported HLE-Diamond partition'):
        _adapter().record_to_sample(_record(0, partition='unexpected'))


def test_report_includes_overall_and_partition_accuracy() -> None:
    adapter = _adapter()
    values = {'reasoning': [1.0, 0.0], 'knowledge': [1.0]}
    score_dict = {
        subset: adapter.aggregate_scores([
            SampleScore(sample_id=index, group_id=index, score=Score(value={'acc': value}))
            for index, value in enumerate(partition_values)
        ])
        for subset, partition_values in values.items()
    }

    report = adapter.generate_report(score_dict, model_name='mock', output_dir='')

    assert report.score == pytest.approx(2 / 3, abs=1e-4)
    subsets = {
        subset.name: subset
        for category in report.primary_metric.categories
        for subset in category.subsets
    }
    assert set(subsets) == {'reasoning', 'knowledge'}
    assert subsets['reasoning'].score == 0.5
    assert subsets['reasoning'].num == 2
    assert subsets['knowledge'].score == 1.0
    assert subsets['knowledge'].num == 1
