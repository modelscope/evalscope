"""A sample may carry no value for a metric (an unparseable LLM judge is excluded rather
than scored 0), so aggregators must never index every sample with ``scores[0]``' keys."""
import pytest
from typing import Any, Dict, List, Optional

from evalscope.api.metric import SampleScore, Score
from evalscope.metrics.aggregators import Mean, MeanPassAtK, MeanPassHatK, MeanVoteAtK

K_AGGREGATORS = [MeanPassAtK, MeanVoteAtK, MeanPassHatK]


def make_sample_score(
    value: Dict[str, float],
    group_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    prediction: Optional[str] = None,
    generation_index: Optional[int] = None,
) -> SampleScore:
    return SampleScore(
        score=Score(value=value, extracted_prediction=prediction),
        sample_id=sample_id or group_id,
        group_id=group_id,
        generation_index=generation_index,
    )


def find_agg(agg_scores: List[Any], aggregation: str, k: int) -> Optional[Any]:
    for agg in agg_scores:
        if agg.aggregation == aggregation and agg.dimensions.get('k') == k:
            return agg
    return None


@pytest.mark.parametrize('aggregator_cls', K_AGGREGATORS)
def test_first_sample_missing_metric_still_reports_it(aggregator_cls):
    scores = [
        make_sample_score({}, group_id='g0', prediction='p'),
        make_sample_score({'judge_score': 1.0}, group_id='g1', prediction='p'),
    ]

    agg_scores = aggregator_cls()(scores)

    assert any(agg.metric_name == 'judge_score' for agg in agg_scores)


@pytest.mark.parametrize('aggregator_cls', K_AGGREGATORS)
def test_later_sample_missing_metric_does_not_raise(aggregator_cls):
    scores = [
        make_sample_score({'judge_score': 1.0}, group_id='g0', prediction='p'),
        make_sample_score({}, group_id='g1', prediction='p'),
    ]

    agg_scores = aggregator_cls()(scores)

    assert any(agg.metric_name == 'judge_score' for agg in agg_scores)


@pytest.mark.parametrize('aggregator_cls', K_AGGREGATORS)
def test_group_with_no_usable_attempt_is_dropped_not_zeroed(aggregator_cls):
    """An unusable judge result must not be averaged in as a 0."""
    scores = [
        make_sample_score({'judge_score': 1.0}, group_id='g0', sample_id='g0-0', prediction='p'),
        make_sample_score({'judge_score': 1.0}, group_id='g0', sample_id='g0-1', prediction='p'),
        make_sample_score({}, group_id='g1', sample_id='g1-0', prediction='p'),
        make_sample_score({}, group_id='g1', sample_id='g1-1', prediction='p'),
    ]

    agg_scores = aggregator_cls()(scores)
    at_1 = [agg for agg in agg_scores if agg.dimensions.get('k') == 1 and agg.metric_name == 'judge_score']

    assert at_1, 'k=1 aggregate is expected'
    assert at_1[0].score == pytest.approx(1.0)
    assert at_1[0].num == 1


def test_pass_at_k_keeps_humaneval_estimator():
    """Locks the combinatorial estimator: a prefix reading would give 0.0 for pass@2 here."""
    values = [0.0, 0.0, 1.0, 1.0]
    scores = [
        make_sample_score({'pass': v}, group_id='g0', sample_id=f'g0-{i}', prediction=str(i))
        for i, v in enumerate(values)
    ]

    agg_scores = MeanPassAtK()(scores)

    assert find_agg(agg_scores, 'pass_at_k', 1).score == pytest.approx(0.5)
    assert find_agg(agg_scores, 'pass_at_k', 2).score == pytest.approx(5 / 6)


def test_pass_hat_k_keeps_estimator():
    """4 attempts, 2 correct: pass^2 = C(2,2)/C(4,2) = 1/6."""
    values = [1.0, 1.0, 0.0, 0.0]
    scores = [
        make_sample_score({'pass': v}, group_id='g0', sample_id=f'g0-{i}', prediction=str(i))
        for i, v in enumerate(values)
    ]

    agg_scores = MeanPassHatK()(scores)

    assert find_agg(agg_scores, 'pass_hat_k', 2).score == pytest.approx(1 / 6)


def test_unusable_attempt_shrinks_n_instead_of_scoring_zero():
    """One unparseable judge result out of 4 leaves a 3-attempt estimator, not a 0."""
    scores = [
        make_sample_score({'judge_score': 1.0}, group_id='g0', sample_id='g0-0', prediction='a'),
        make_sample_score({'judge_score': 0.0}, group_id='g0', sample_id='g0-1', prediction='b'),
        make_sample_score({'judge_score': 0.0}, group_id='g0', sample_id='g0-2', prediction='c'),
        make_sample_score({}, group_id='g0', sample_id='g0-3', prediction='d'),
    ]

    agg_scores = MeanPassAtK()(scores)

    # n=3, c=1 -> pass@1 = 1/3; a 0-scored 4th attempt would have given 1/4.
    assert find_agg(agg_scores, 'pass_at_k', 1).score == pytest.approx(1 / 3)
    assert find_agg(agg_scores, 'pass_at_k', 4) is None


def test_mean_only_counts_present_values():
    scores = [
        make_sample_score({'judge_score': 1.0}, group_id='g0'),
        make_sample_score({}, group_id='g1'),
    ]

    agg_scores = Mean()(scores)

    assert len(agg_scores) == 1
    assert agg_scores[0].score == pytest.approx(1.0)
    assert agg_scores[0].num == 1


@pytest.mark.parametrize('aggregator_cls', K_AGGREGATORS)
def test_missing_first_generation_never_promotes_a_later_generation_to_k_one(aggregator_cls):
    scores = [
        make_sample_score({'judge_score': 1.0}, group_id='g0', sample_id='g0-1', prediction='right', generation_index=1),
        make_sample_score({'judge_score': 0.0}, group_id='g1', sample_id='g1-0', prediction='wrong', generation_index=0),
    ]

    at_one = find_agg(aggregator_cls()(scores),
                      'vote_at_k' if aggregator_cls is MeanVoteAtK else
                      'pass_at_k' if aggregator_cls is MeanPassAtK else 'pass_hat_k', 1)

    assert at_one.num == 1
    assert at_one.metadata['excluded'] == 1


@pytest.mark.parametrize('aggregator_cls', K_AGGREGATORS)
def test_missing_middle_generation_excludes_only_affected_higher_k_groups(aggregator_cls):
    scores = [
        make_sample_score({'judge_score': 1.0}, group_id='g0', sample_id='g0-0', generation_index=0),
        make_sample_score({}, group_id='g0', sample_id='g0-1', generation_index=1),
        make_sample_score({'judge_score': 1.0}, group_id='g0', sample_id='g0-2', generation_index=2),
        make_sample_score({'judge_score': 0.0}, group_id='g1', sample_id='g1-0', generation_index=0),
        make_sample_score({'judge_score': 0.0}, group_id='g1', sample_id='g1-1', generation_index=1),
        make_sample_score({'judge_score': 0.0}, group_id='g1', sample_id='g1-2', generation_index=2),
    ]

    aggregation = (
        'vote_at_k' if aggregator_cls is MeanVoteAtK else
        'pass_at_k' if aggregator_cls is MeanPassAtK else 'pass_hat_k'
    )
    at_two = find_agg(aggregator_cls()(scores), aggregation, 2)

    assert at_two is not None
    assert at_two.num == 1
    assert at_two.metadata == {'eligible': 1, 'total': 2, 'coverage': 0.5, 'excluded': 1}
