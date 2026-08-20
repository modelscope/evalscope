"""``scoring_policy`` decides which judge strategies a benchmark can honour."""
import pytest

from evalscope.api.mixin import LLMJudgeMixin
from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import JudgeStrategy, ScoringPolicy

JUDGE_ARGS = {'model_id': 'judge-model'}


def test_policy_derives_both_orthogonal_facts():
    assert ScoringPolicy.RULE_DEFAULT.rule_supported
    assert not ScoringPolicy.RULE_DEFAULT.judge_by_default

    assert ScoringPolicy.JUDGE_DEFAULT.rule_supported
    assert ScoringPolicy.JUDGE_DEFAULT.judge_by_default

    assert not ScoringPolicy.JUDGE_ONLY.rule_supported
    assert ScoringPolicy.JUDGE_ONLY.judge_by_default


def test_default_policy_is_rule_default():
    assert LLMJudgeMixin.scoring_policy is ScoringPolicy.RULE_DEFAULT


def test_legacy_llm_judge_default_maps_conservatively():
    """``True`` must map to JUDGE_DEFAULT, not JUDGE_ONLY, so third-party adapters keep rule mode."""

    class LegacyTrue(LLMJudgeMixin):
        llm_judge_default = True

    class LegacyFalse(LLMJudgeMixin):
        llm_judge_default = False

    class Explicit(LLMJudgeMixin):
        llm_judge_default = True
        scoring_policy = ScoringPolicy.JUDGE_ONLY

    assert LegacyTrue.scoring_policy is ScoringPolicy.JUDGE_DEFAULT
    assert LegacyFalse.scoring_policy is ScoringPolicy.RULE_DEFAULT
    assert Explicit.scoring_policy is ScoringPolicy.JUDGE_ONLY


@pytest.mark.parametrize('strategy', [JudgeStrategy.RULE, JudgeStrategy.LLM_RECALL])
def test_judge_only_benchmark_rejects_rule_dependent_strategies(strategy):
    cfg = TaskConfig(model='m', datasets=['simple_qa'], judge={'strategy': strategy, 'models': JUDGE_ARGS})

    adapter = get_benchmark('simple_qa', cfg)
    with pytest.raises(ValueError, match='no usable rule-based scoring'):
        adapter.validate_judge_strategy()


def test_judge_only_benchmark_accepts_llm_and_auto():
    for strategy in (JudgeStrategy.LLM, JudgeStrategy.AUTO):
        cfg = TaskConfig(model='m', datasets=['simple_qa'], judge={'strategy': strategy, 'models': JUDGE_ARGS})
        assert get_benchmark('simple_qa', cfg).use_llm_judge


def test_judge_default_benchmark_still_allows_rule():
    cfg = TaskConfig(model='m', datasets=['minerva_math'], judge={'strategy': JudgeStrategy.RULE})

    adapter = get_benchmark('minerva_math', cfg)

    assert adapter.scoring_policy is ScoringPolicy.JUDGE_DEFAULT
    assert not adapter.use_llm_judge


def test_judge_default_benchmark_uses_judge_under_auto():
    cfg = TaskConfig(model='m', datasets=['minerva_math'], judge={'models': JUDGE_ARGS})

    assert get_benchmark('minerva_math', cfg).use_llm_judge


def test_rule_default_benchmark_scores_by_rule_under_auto():
    cfg = TaskConfig(model='m', datasets=['gsm8k'])

    adapter = get_benchmark('gsm8k', cfg)

    assert adapter.scoring_policy is ScoringPolicy.RULE_DEFAULT
    assert not adapter.use_llm_judge


def test_missing_judge_models_fails_before_generating():
    cfg = TaskConfig(model='m', datasets=['simple_qa'], judge={'strategy': JudgeStrategy.LLM})

    adapter = get_benchmark('simple_qa', cfg)
    with pytest.raises(ValueError, match='judge.models must be provided'):
        adapter.validate_judge_strategy()


def test_metadata_construction_never_validates_judge_configuration():
    cfg = TaskConfig(model='m', datasets=['simple_qa'], judge={'strategy': JudgeStrategy.RULE})

    assert get_benchmark('simple_qa', cfg) is not None


def test_every_benchmark_declares_a_known_policy():
    for name, meta in BENCHMARK_REGISTRY.items():
        policy = meta.data_adapter.scoring_policy
        assert isinstance(policy, ScoringPolicy), f'{name} declares a non-ScoringPolicy value: {policy!r}'
