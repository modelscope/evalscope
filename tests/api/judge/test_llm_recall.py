"""``llm_recall`` recovers rule-based misses, so the judge can only raise the score.

A judge that failed must leave the rule score intact: erasing rule evidence was the defect that
came from scoring ``[ERROR]`` responses as 0.
"""
import pytest

from evalscope.api.metric import JudgeSummary, Score
from evalscope.api.mixin import LLMJudgeMixin
from evalscope.constants import ScoreStatus


class Merger(LLMJudgeMixin):
    """Exercises ``_merge_scores`` without needing a benchmark or a judge model."""

    def __init__(self) -> None:
        pass


@pytest.mark.parametrize(
    'rule_value, judge_value, expected',
    [
        (0.0, 1.0, 1.0),  # the judge recovers a rule miss
        (0.0, 0.0, 0.0),
        (0.6, 0.2, 0.6),  # a continuous rule score is not lowered by the judge
        (0.4, 0.9, 0.9),
    ],
)
def test_merge_takes_the_maximum(rule_value, judge_value, expected):
    rule = Score(value={'acc': rule_value}, main_score_name='acc')
    judge = Score(value={'acc': judge_value}, main_score_name='acc')

    merged = Merger()._merge_scores(rule, judge)

    assert merged.value['acc'] == expected
    assert merged.status is ScoreStatus.SUCCESS


@pytest.mark.parametrize('status', [ScoreStatus.TRANSPORT_ERROR, ScoreStatus.PARSE_ERROR, ScoreStatus.EXCLUDED])
def test_an_unusable_judge_keeps_the_rule_score(status):
    rule = Score(value={'acc': 0.6}, main_score_name='acc')
    judge = Score(value={}, status=status, judge_summary=JudgeSummary(judge_models=['j']))

    merged = Merger()._merge_scores(rule, judge)

    assert merged.value == {'acc': 0.6}
    assert merged.status is ScoreStatus.FALLBACK
    assert merged.status.is_usable, 'a rule-scored sample must still count towards the metric'
    assert merged.metadata['judge_unavailable'] == status.value
    assert merged.judge_summary.judge_models == ['j']
    assert merged.judge_summary.status is ScoreStatus.FALLBACK


def test_a_judge_fallback_score_is_still_merged():
    rule = Score(value={'acc': 0.0}, main_score_name='acc')
    judge = Score(value={'acc': 1.0}, status=ScoreStatus.FALLBACK, main_score_name='acc')

    merged = Merger()._merge_scores(rule, judge)

    assert merged.value['acc'] == 1.0
    assert merged.status is ScoreStatus.FALLBACK
