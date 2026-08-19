"""Judge configuration is typed before the Native judge subsystem sees it."""
import pytest

from evalscope.config import JudgeConfig, TaskConfig


def test_typed_config_assigns_unique_model_id_as_judge_id():
    config = JudgeConfig(models=[{'model_id': 'judge-a'}])

    assert config.models[0].judge_id == 'judge-a'


def test_duplicate_model_id_requires_explicit_judge_ids():
    with pytest.raises(ValueError, match='needs judge_id'):
        JudgeConfig(models=[{'model_id': 'same'}, {'model_id': 'same'}])


def test_duplicate_model_id_is_allowed_with_distinct_judge_ids():
    config = JudgeConfig(models=[{'model_id': 'same', 'judge_id': 'a'}, {'model_id': 'same', 'judge_id': 'b'}])

    assert [model.judge_id for model in config.models] == ['a', 'b']


def test_new_config_exposes_repeats_swap_aggregation_and_quorum():
    config = TaskConfig(
        model='m',
        datasets=['simple_qa'],
        judge={
            'strategy': 'llm',
            'models': [{'model_id': 'judge-a'}],
            'repeats': 2,
            'position_swap': 'on',
            'aggregation': 'median',
            'min_valid_judges': 1,
        },
    )

    assert config.judge.repeats == 2
    assert config.judge.position_swap == 'on'
    assert config.judge.aggregation == 'median'


def test_legacy_single_mapping_is_converted_at_the_boundary():
    config = TaskConfig(
        model='m',
        datasets=['simple_qa'],
        judge_strategy='llm',
        judge_model_args={'model_id': 'judge-a'},
    )

    assert config.judge.strategy == 'llm'
    assert config.judge.models[0].model_id == 'judge-a'
    assert 'judge_model_args' not in config.model_dump()


def test_pr_only_list_legacy_shape_is_rejected():
    with pytest.raises(ValueError, match='list-valued'):
        TaskConfig(model='m', datasets=['simple_qa'], judge_model_args=[{'model_id': 'judge-a'}])


def test_pr_only_judge_repeats_alias_is_rejected():
    with pytest.raises(ValueError, match='judge_repeats'):
        TaskConfig(model='m', datasets=['simple_qa'], judge_repeats=2)
