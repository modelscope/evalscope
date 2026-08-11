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
