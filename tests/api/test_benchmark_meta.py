"""Tests for benchmark metadata validation."""

import json

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
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


def test_few_shot_metadata_accepts_auto_without_a_train_split() -> None:
    meta = BenchmarkMeta(name='auto_without_train_split', dataset_id='local', few_shot_num=1)

    assert meta.few_shot_mode == 'auto'

    with pytest.raises(ValueError, match='fixed few_shot_mode requires allowed_few_shot_nums'):
        BenchmarkMeta(name='invalid_fixed', dataset_id='local', few_shot_mode='fixed')


def test_few_shot_capabilities_cannot_be_overridden_at_runtime() -> None:
    meta = BenchmarkMeta(name='protected_few_shot_mode', dataset_id='local')

    with pytest.raises(ValueError, match='few_shot_mode'):
        meta._update({'few_shot_mode': 'disabled'})


@pytest.mark.parametrize('benchmark_name', ['longbench_v2', 'agieval', 'coin_flip', 'general_vmcq', 'zerobench'])
def test_unsupported_benchmarks_reject_few_shot_before_loading_datasets(benchmark_name: str) -> None:
    with pytest.raises(ValueError, match=f'Benchmark {benchmark_name!r} does not support few-shot'):
        get_benchmark(
            benchmark_name,
            TaskConfig(
                model='mock',
                datasets=[benchmark_name],
                dataset_args={benchmark_name: {'few_shot_num': 1}},
            ),
        )


def test_benchmark_few_shot_modes_validate_before_loading_datasets() -> None:

    fixed_adapter = get_benchmark(
        'gpqa_diamond',
        TaskConfig(model='mock', datasets=['gpqa_diamond'], dataset_args={'gpqa_diamond': {'few_shot_num': 5}}),
    )
    assert fixed_adapter.few_shot_mode == 'fixed'
    assert not fixed_adapter._should_load_fewshot()

    auto_adapter = get_benchmark(
        'mmlu',
        TaskConfig(model='mock', datasets=['mmlu'], dataset_args={'mmlu': {'few_shot_num': 2}}),
    )
    assert auto_adapter.few_shot_mode == 'auto'
    assert auto_adapter._should_load_fewshot()

    with pytest.raises(ValueError, match='supports few_shot_num values: 0, 5; got 2'):
        get_benchmark(
            'gpqa_diamond',
            TaskConfig(model='mock', datasets=['gpqa_diamond'], dataset_args={'gpqa_diamond': {'few_shot_num': 2}}),
        )

    with pytest.raises(ValueError, match='supports few_shot_num values: 0, 1, 2, 3; got 4'):
        get_benchmark(
            'race',
            TaskConfig(model='mock', datasets=['race'], dataset_args={'race': {'few_shot_num': 4}}),
        )


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
    assert 'few_shot_mode' not in extracted
    assert 'allowed_few_shot_nums' not in extracted
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
