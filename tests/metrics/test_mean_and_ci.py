"""Tests for the mean_and_ci aggregator (issue: report score confidence intervals).

Wilson intervals are verified against closed-form values computed independently
(python, z=1.959964): p=0.882, n=500 -> [0.8510, 0.9074]; p=0.85, n=20 -> [0.6397, 0.9476].
"""
import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.metrics.aggregators.aggregators import Mean, MeanAndCI


def ratio_scores(ones: int, zeros: int, prefix: str = 's'):
    values = [1.0] * ones + [0.0] * zeros
    return [
        SampleScore(sample_id=f'{prefix}-{i}', score=Score(value={'accuracy': values[i]}))
        for i in range(ones + zeros)
    ]


def continuous_scores(values, prefix: str = 'c'):
    # 'score' is a reserved metric name (canonicalizes to legacy_metric); use a legal one.
    return [
        SampleScore(sample_id=f'{prefix}-{i}', score=Score(value={'judge_score': v}))
        for i, v in enumerate(values)
    ]


def find(entries, aggregation, metric='accuracy', method=None):
    for entry in entries:
        if entry.aggregation != aggregation or entry.metric_name != metric:
            continue
        if method is not None and (entry.metadata or {}).get('ci_method') != method:
            continue
        return entry
    return None


def test_wilson_interval_matches_closed_form():
    scores = ratio_scores(ones=441, zeros=59)  # mean 0.882, n=500
    ci = find(MeanAndCI()(scores), 'mean_and_ci')

    assert ci.metadata['ci_low'] == pytest.approx(0.8510, abs=0.002)
    assert ci.metadata['ci_high'] == pytest.approx(0.9074, abs=0.002)
    assert ci.metadata['ci_method'] == 'wilson'
    assert ci.metadata['ci_level'] == 0.95


def test_plain_mean_is_emitted_alongside():
    entries = MeanAndCI()(ratio_scores(ones=441, zeros=59))
    plain = find(entries, 'mean')
    ci = find(entries, 'mean_and_ci')

    assert plain is not None and ci is not None
    assert plain.aggregation == 'mean'
    assert plain.score == ci.score
    assert plain.num == ci.num == 500
    assert plain.num_correct == ci.num_correct == 441
    assert ci.dimensions == {'method': 'ci'}
    assert plain.metadata is None and ci.metadata.get('ci_low') is not None


def test_output_count_is_two_per_metric():
    assert len(MeanAndCI()(ratio_scores(ones=8, zeros=2))) == 2


def test_continuous_metric_uses_bootstrap_and_is_deterministic():
    values = [0.5 + 0.01 * (i % 50) for i in range(300)]
    first = find(MeanAndCI()(continuous_scores(values)), 'mean_and_ci', metric='judge_score')
    second = find(MeanAndCI()(continuous_scores(values)), 'mean_and_ci', metric='judge_score')

    assert first.metadata['ci_method'] == 'bootstrap'
    assert first.metadata['ci_low'] == second.metadata['ci_low']
    assert first.metadata['ci_high'] == second.metadata['ci_high']
    assert first.metadata['ci_low'] < first.score < first.metadata['ci_high']


def test_small_ratio_subset_flags_both_conditions():
    scores = ratio_scores(ones=17, zeros=3)  # mean 0.85, n=20
    ci = find(MeanAndCI()(scores), 'mean_and_ci')

    flags = ci.metadata['flags']
    assert 'low_confidence' in flags
    assert 'wide_interval' in flags


def test_large_narrow_subset_has_no_flags():
    scores = ratio_scores(ones=441, zeros=59)  # width ~5.6pt, n=500
    ci = find(MeanAndCI()(scores), 'mean_and_ci')

    assert ci.metadata['flags'] == []


def test_empty_input_returns_empty_list():
    assert MeanAndCI()([]) == []


def test_plain_mean_matches_mean_aggregator_exactly():
    scores = ratio_scores(ones=7, zeros=3)
    mean_entry = Mean()(scores)[0]
    plain_from_ci = find(MeanAndCI()(scores), 'mean')

    assert plain_from_ci.score == mean_entry.score
    assert plain_from_ci.num == mean_entry.num
    assert plain_from_ci.num_correct == mean_entry.num_correct
