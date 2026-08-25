"""Judge configuration is typed before the Native judge subsystem sees it."""
import argparse

import pytest

from evalscope.arguments import add_argument
from evalscope.config import JudgeConfig, TaskConfig
from evalscope.metrics.judge.llm_judge import LLMJudge


def test_typed_config_assigns_unique_model_id_as_judge_id():
    config = JudgeConfig(models=[{'model_id': 'judge-a'}])

    assert config.models[0].judge_id == 'judge-a'


def test_typed_config_accepts_a_single_model_mapping():
    config = JudgeConfig(models={'model_id': 'judge-a'})

    assert [model.model_id for model in config.models] == ['judge-a']


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


def test_task_config_accepts_a_single_judge_model_mapping():
    config = TaskConfig(
        model='m',
        datasets=['simple_qa'],
        judge={'strategy': 'llm', 'models': {'model_id': 'judge-a'}},
    )

    assert [model.model_id for model in config.judge.models] == ['judge-a']


def test_update_revalidates_judge_config():
    config = TaskConfig(model='m', datasets=['simple_qa'])

    config.update({'judge': {'strategy': 'llm', 'models': {'model_id': 'judge-a'}}})

    assert isinstance(config.judge, JudgeConfig)
    assert [model.model_id for model in config.judge.models] == ['judge-a']


def test_cli_parses_typed_judge_config_without_removed_legacy_defaults():
    parser = argparse.ArgumentParser()
    add_argument(parser)

    args = parser.parse_args([
        '--model', 'm', '--datasets', 'simple_qa', '--judge', '{"strategy": "llm", "models": {"model_id": "judge-a"}}'
    ])
    config = TaskConfig.from_args(args)

    assert config.judge.strategy == 'llm'
    assert [model.model_id for model in config.judge.models] == ['judge-a']


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


def test_legacy_semantics_move_to_the_shared_contract():
    config = TaskConfig(
        model='m',
        datasets=['simple_qa'],
        judge_strategy='llm',
        judge_model_args={
            'model_id': 'judge-a',
            'prompt_template': 'Question: {question}',
            'score_mapping': {'yes': 1.0, 'no': 0.0},
        },
    )

    assert config.judge.contract.prompt_template == 'Question: {question}'
    assert config.judge.contract.score_mapping == {'yes': 1.0, 'no': 0.0}
    assert 'prompt_template' not in config.judge.models[0].model_dump()


def test_model_specific_semantics_are_rejected():
    with pytest.raises(ValueError, match='prompt_template'):
        JudgeConfig(models=[{'model_id': 'judge-a', 'prompt_template': 'not allowed'}])


def test_removed_judge_worker_num_is_rejected():
    with pytest.raises(ValueError, match='judge_worker_num'):
        TaskConfig(model='m', datasets=['simple_qa'], judge_worker_num=1)


@pytest.mark.parametrize('value', [
    {'strategy': 'not-a-strategy'},
    {'contract': {'score_type': 'not-a-contract'}},
    {'contract': {'score_typo': 'pattern'}},
    {'min_valid_judge': 2},
])
def test_typed_judge_config_rejects_unknown_or_invalid_values(value):
    with pytest.raises(ValueError):
        JudgeConfig(**value)


def test_legacy_score_pattern_has_an_actionable_error():
    with pytest.raises(ValueError, match='score_pattern'):
        TaskConfig(
            model='m',
            datasets=['simple_qa'],
            judge_model_args={'model_id': 'judge-a', 'score_pattern': '[[A]]'},
        )


def test_llm_judge_is_concrete_after_removing_the_legacy_judge_api(monkeypatch):
    monkeypatch.setattr(LLMJudge, '_init_server_adapter', lambda self: None)

    judge = LLMJudge(model_id='judge-a')

    assert judge.model_id == 'judge-a'


def test_llm_judge_rejects_the_removed_score_pattern_parameter():
    with pytest.raises(TypeError, match='score_pattern'):
        LLMJudge(model_id='judge-a', score_pattern='[[A]]')
