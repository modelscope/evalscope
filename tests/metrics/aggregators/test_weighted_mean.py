"""Instruction-level metrics are per-sample ratios, so averaging the ratios makes a prompt with one
instruction count as much as a prompt with three. ``WeightedMean`` restores the official
micro-average by pooling the underlying units."""
from typing import Dict, List, Optional

import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.metrics.aggregators import METRIC_WEIGHTS_KEY, WeightedMean


def make_sample_score(
    value: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    sample_id: Optional[str] = None,
) -> SampleScore:
    metadata = {METRIC_WEIGHTS_KEY: weights} if weights is not None else {}
    return SampleScore(score=Score(value=value, metadata=metadata), sample_id=sample_id)


def find(agg_scores: List, metric_name: str):
    for agg in agg_scores:
        if agg.metric_name == metric_name:
            return agg
    return None


def test_weighted_metric_pools_units_instead_of_averaging_samples():
    """One instruction followed out of a 1-inst prompt and a 3-inst prompt: 2/4, not (1 + 1/3)/2."""
    scores = [
        make_sample_score({'inst_level_strict': 1.0}, {'inst_level_strict': 1}, sample_id='s0'),
        make_sample_score({'inst_level_strict': 1 / 3}, {'inst_level_strict': 3}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'inst_level_strict')

    assert agg.score == pytest.approx(0.5)
    assert agg.score != pytest.approx(2 / 3), 'macro-average leaked through'


def test_num_reports_total_weight_so_cross_subset_rollup_stays_micro():
    """``Subset.num`` feeds the report layer's ``micro_mean``; for a weighted metric it must be the
    unit count, otherwise a multi-subset benchmark re-introduces the macro bias one level up."""
    scores = [
        make_sample_score({'inst_level_strict': 1.0}, {'inst_level_strict': 1}, sample_id='s0'),
        make_sample_score({'inst_level_strict': 1 / 3}, {'inst_level_strict': 3}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'inst_level_strict')

    assert agg.num == 4
    assert agg.metadata['samples'] == 2
    assert agg.metadata['weighted'] is True


def test_unweighted_metric_keeps_plain_mean_and_sample_count():
    """prompt-level accuracy is already one unit per sample; weighting must not touch it."""
    scores = [
        make_sample_score({'prompt_level_strict': 1.0}, {'inst_level_strict': 1}, sample_id='s0'),
        make_sample_score({'prompt_level_strict': 0.0}, {'inst_level_strict': 3}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'prompt_level_strict')

    assert agg.score == pytest.approx(0.5)
    assert agg.num == 2
    assert agg.metadata['weighted'] is False


def test_weighted_and_unweighted_metrics_coexist_in_one_score():
    scores = [
        make_sample_score(
            {'prompt_level_strict': 1.0, 'inst_level_strict': 1.0},
            {'inst_level_strict': 1},
            sample_id='s0',
        ),
        make_sample_score(
            {'prompt_level_strict': 0.0, 'inst_level_strict': 1 / 3},
            {'inst_level_strict': 3},
            sample_id='s1',
        ),
    ]

    agg_scores = WeightedMean()(scores)

    assert find(agg_scores, 'prompt_level_strict').score == pytest.approx(0.5)
    assert find(agg_scores, 'inst_level_strict').score == pytest.approx(0.5)


def test_scores_without_declared_weights_fall_back_to_plain_mean():
    """Any benchmark may select this aggregator without emitting weights."""
    scores = [
        make_sample_score({'accuracy': 1.0}, sample_id='s0'),
        make_sample_score({'accuracy': 0.0}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'accuracy')

    assert agg.score == pytest.approx(0.5)
    assert agg.num == 2


def test_only_counts_present_values():
    """A checker failure empties ``Score.value``; that sample must drop out, not score 0."""
    scores = [
        make_sample_score({'inst_level_strict': 1.0}, {'inst_level_strict': 2}, sample_id='s0'),
        make_sample_score({}, sample_id='s1'),
    ]

    agg_scores = WeightedMean()(scores)

    assert len(agg_scores) == 1
    assert agg_scores[0].score == pytest.approx(1.0)
    assert agg_scores[0].num == 2


def test_zero_weight_metric_falls_back_instead_of_dividing_by_zero():
    """A prompt with an empty instruction list carries no units to pool."""
    scores = [
        make_sample_score({'inst_level_strict': 0.0}, {'inst_level_strict': 0}, sample_id='s0'),
        make_sample_score({'inst_level_strict': 1.0}, {'inst_level_strict': 0}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'inst_level_strict')

    assert agg.score == pytest.approx(0.5)


def test_malformed_weight_metadata_is_ignored():
    scores = [
        SampleScore(
            score=Score(value={'inst_level_strict': 1.0}, metadata={METRIC_WEIGHTS_KEY: 'not-a-dict'}),
            sample_id='s0',
        ),
        make_sample_score({'inst_level_strict': 0.0}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'inst_level_strict')

    assert agg.score == pytest.approx(0.5)


def test_all_single_unit_weights_still_count_as_weighted():
    """Declared-ness decides, not the numeric value: a dataset of 1-instruction prompts is still a
    weighted metric, and ``num`` is a unit total that merely coincides with the sample count."""
    scores = [
        make_sample_score({'inst_level_strict': 1.0}, {'inst_level_strict': 1}, sample_id='s0'),
        make_sample_score({'inst_level_strict': 0.0}, {'inst_level_strict': 1}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'inst_level_strict')

    assert agg.score == pytest.approx(0.5)
    assert agg.num == 2
    assert agg.metadata['weighted'] is True


def test_prompt_with_no_instructions_is_excluded_rather_than_scored_zero():
    """``agg_inst_level_acc`` returns 0 for an empty instruction list; pooling units must not let
    that spurious 0 drag the instruction-level score down."""
    scores = [
        make_sample_score({'inst_level_strict': 0.0}, {'inst_level_strict': 0}, sample_id='empty'),
        make_sample_score({'inst_level_strict': 1.0}, {'inst_level_strict': 2}, sample_id='s1'),
    ]

    agg = find(WeightedMean()(scores), 'inst_level_strict')

    assert agg.score == pytest.approx(1.0)
    assert agg.num == 2


def test_empty_scores_return_no_aggregates():
    assert WeightedMean()([]) == []


def test_identity_carries_the_weighted_mean_aggregation_name():
    """The report layer keys semantics off this name, so it must not silently stay 'mean'."""
    scores = [make_sample_score({'inst_level_strict': 1.0}, {'inst_level_strict': 2}, sample_id='s0')]

    assert find(WeightedMean()(scores), 'inst_level_strict').aggregation == 'weighted_mean'
