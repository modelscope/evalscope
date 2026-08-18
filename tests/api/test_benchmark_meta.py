"""Tests for benchmark metadata validation."""

import json
import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.metric.semantics import MetricSelector
from evalscope.utils.doc_utils.generate_dataset_md import extract_benchmark_meta


def test_runtime_update_revalidates_primary_metric() -> None:
    meta = BenchmarkMeta(
        name='multi_metric',
        dataset_id='local',
        metric_list=['accuracy', 'f1_score'],
        primary_metric='accuracy',
    )

    with pytest.raises(ValueError, match="primary_metric='missing'"):
        meta._update({'primary_metric': 'missing'})


def test_string_primary_metric_is_first_class_shorthand() -> None:
    meta = BenchmarkMeta(
        name='single_name_selector',
        dataset_id='local',
        metric_list=['accuracy'],
        primary_metric='accuracy',
    )

    assert meta.primary_metric == MetricSelector(name='accuracy')


def test_legacy_metric_list_aliases_are_normalized_at_the_adapter_boundary() -> None:
    meta = BenchmarkMeta(
        name='legacy_adapter',
        dataset_id='local',
        metric_list=['acc', 'f1_score'],
        primary_metric=MetricSelector(name='accuracy'),
    )

    assert meta.metric_list == ['accuracy', 'f1']


def test_runtime_metric_list_update_revalidates_primary_metric() -> None:
    meta = BenchmarkMeta(
        name='multi_metric',
        dataset_id='local',
        metric_list=['accuracy', 'f1_score'],
        primary_metric='accuracy',
    )

    with pytest.raises(ValueError, match="primary_metric='accuracy'"):
        meta._update({'metric_list': ['f1_score']})


def test_doc_metadata_extraction_does_not_instantiate_adapter() -> None:

    class RuntimeOnlyAdapter:

        def __init__(self) -> None:
            raise AssertionError('documentation metadata must not instantiate the adapter')

    meta = BenchmarkMeta(
        name='runtime_only',
        dataset_id='local',
        metric_list=['accuracy'],
        data_adapter=RuntimeOnlyAdapter,
    )

    extracted = extract_benchmark_meta(meta, RuntimeOnlyAdapter)

    assert extracted['metrics'] == ['accuracy']
    assert 'primary_metric' not in extracted
    assert extracted['category'] == 'llm'


def test_doc_metadata_serializes_structured_primary_metric() -> None:
    meta = BenchmarkMeta(
        name='structured_primary',
        dataset_id='local',
        metric_list=['accuracy'],
        primary_metric=MetricSelector(name='accuracy', aggregation='pass_at_k', dimensions={'k': 1}),
    )

    extracted = extract_benchmark_meta(meta, None)

    assert extracted['primary_metric'] == {
        'name': 'accuracy',
        'aggregation': 'pass_at_k',
        'dimensions': {
            'k': 1
        },
    }
    json.dumps(extracted)


def test_adapter_init_preserves_the_declared_shuffle_choices() -> None:
    """`DataAdapter.__init__` used to assign `self.shuffle_choices = False`.

    That runs through the property setter and writes into the meta, so it silently discarded both
    the benchmark's declaration and any `dataset_args` value, leaving the switch unreachable.
    """
    from evalscope.api.registry import get_benchmark

    adapter = get_benchmark('truthful_qa', validate_judge=False)

    assert adapter.shuffle_choices is True


def test_dataset_args_can_turn_shuffle_choices_off() -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    config = TaskConfig(datasets=['truthful_qa'], dataset_args={'truthful_qa': {'shuffle_choices': False}})
    adapter = get_benchmark('truthful_qa', config=config, validate_judge=False)

    assert adapter.shuffle_choices is False
