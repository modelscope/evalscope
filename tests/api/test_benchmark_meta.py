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


def test_dataset_args_survives_adapter_init_across_benchmarks() -> None:
    """An adapter must not overwrite a field the user set through `dataset_args`.

    Assigning a meta-backed field in `__init__` writes through the property setter, and
    `dataset_args` is merged before the adapter is built, so an unguarded assignment discards the
    user's value silently. `truthful_qa`, `general_mcq` and `scicode` each did this.
    """
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    custom = 'CUSTOM TEMPLATE {question} {choices}'
    for name in ('truthful_qa', 'general_mcq', 'scicode', 'mmlu'):
        config = TaskConfig(datasets=[name], dataset_args={name: {'prompt_template': custom}})
        adapter = get_benchmark(name, config=config, validate_judge=False)

        assert adapter.prompt_template == custom, f'{name} discarded the configured prompt_template'


def test_benchmark_defaults_still_apply_without_dataset_args() -> None:
    from evalscope.api.registry import get_benchmark

    adapter = get_benchmark('truthful_qa', validate_judge=False)

    assert adapter.prompt_template is not None
    assert not adapter.is_user_configured('prompt_template')


def test_user_overrides_stay_out_of_the_serialized_meta() -> None:
    """The tracking attribute must not leak into `to_dict`, which is dumped into the task config."""
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    config = TaskConfig(datasets=['mmlu'], dataset_args={'mmlu': {'few_shot_num': 3}})
    adapter = get_benchmark('mmlu', config=config, validate_judge=False)

    assert adapter.is_user_configured('few_shot_num')
    assert '_user_overrides' not in adapter.to_dict()
