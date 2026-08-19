"""Judge configuration is validated before any sample is generated.

``judge_model_args`` accepts one mapping or a list of them; ``judge_repeats`` above 1 needs a
non-deterministic judge to be worth its cost.
"""
import pytest

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

JUDGE = {'model_id': 'judge-a', 'generation_config': {'temperature': 0.0}}
HOT_JUDGE = {'model_id': 'judge-a', 'generation_config': {'temperature': 0.7}}


def make_adapter(**overrides):
    config = TaskConfig(model='m', datasets=['simple_qa'], judge_strategy='llm', **overrides)
    return get_benchmark('simple_qa', config, validate_judge=False)


def test_a_single_mapping_is_one_judge_spec():
    adapter = make_adapter(judge_model_args=JUDGE)

    assert adapter._judge_specs() == [JUDGE]


def test_a_list_is_kept_as_several_judge_specs():
    adapter = make_adapter(judge_model_args=[JUDGE, {'model_id': 'judge-b'}])

    assert adapter._judge_specs() == [JUDGE, {'model_id': 'judge-b'}]


def test_no_judge_args_is_no_spec():
    adapter = make_adapter(judge_model_args={})

    assert adapter._judge_specs() == []


def test_repeats_default_to_one():
    assert TaskConfig(model='m', datasets=['simple_qa']).judge_repeats == 1


def test_repeats_below_one_are_rejected():
    with pytest.raises(ValueError):
        TaskConfig(model='m', datasets=['simple_qa'], judge_repeats=0)


def test_repeats_on_a_deterministic_judge_are_rejected():
    """Repeating a temperature-0 judge multiplies cost without adding information."""
    adapter = make_adapter(judge_model_args=JUDGE, judge_repeats=3)

    with pytest.raises(ValueError, match='non-zero judge temperature'):
        adapter.validate_judge_strategy()


def test_repeats_without_an_explicit_temperature_are_rejected():
    adapter = make_adapter(judge_model_args={'model_id': 'judge-a'}, judge_repeats=3)

    with pytest.raises(ValueError, match='non-zero judge temperature'):
        adapter.validate_judge_strategy()


def test_repeats_on_a_sampling_judge_are_accepted():
    adapter = make_adapter(judge_model_args=HOT_JUDGE, judge_repeats=3)

    adapter.validate_judge_strategy()


def test_every_listed_judge_must_allow_repeats():
    adapter = make_adapter(judge_model_args=[HOT_JUDGE, {'model_id': 'judge-b'}], judge_repeats=2)

    with pytest.raises(ValueError, match='non-zero judge temperature'):
        adapter.validate_judge_strategy()


def test_one_repeat_ignores_the_temperature():
    adapter = make_adapter(judge_model_args=JUDGE, judge_repeats=1)

    adapter.validate_judge_strategy()


def test_duplicate_judge_model_ids_are_rejected():
    """Verdicts aggregate per judge id, so a shared id would silently merge two judges."""
    adapter = make_adapter(judge_model_args=[JUDGE, dict(JUDGE)])

    with pytest.raises(ValueError, match='duplicate judge model_id'):
        adapter.init_llm_judges()


def test_judges_disagreeing_on_the_contract_are_rejected():
    """One request shape is built per sample, so a differently configured judge would be graded
    against the primary judge's contract."""
    adapter = make_adapter(
        judge_model_args=[JUDGE, {
            'model_id': 'judge-b',
            'score_type': 'numeric'
        }],
    )

    with pytest.raises(ValueError, match='same score_type'):
        adapter.init_llm_judges()


def test_judges_agreeing_on_the_contract_are_accepted():
    adapter = make_adapter(judge_model_args=[JUDGE, {'model_id': 'judge-b'}])

    assert [judge.model_id for judge in adapter.init_llm_judges()] == ['judge-a', 'judge-b']
