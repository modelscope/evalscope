"""Tests for the mean_and_ci aggregator (issue: report score confidence intervals).

Wilson intervals are verified against closed-form values computed independently
(python, z=1.959964): p=0.882, n=500 -> [0.8510, 0.9074]; p=0.85, n=20 -> [0.6397, 0.9476].

v2 protocol: one AggScore per metric (`aggregation='mean'` for downstream compatibility),
with the interval stored losslessly in `metadata['ci']` and formatted at the display boundary.
"""
import pytest

from evalscope.api.metric import MetricIdentity, SampleScore, Score
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


def find(entries, metric='accuracy'):
    for entry in entries:
        if entry.metric_name == metric:
            return entry
    return None


def test_wilson_interval_matches_closed_form():
    scores = ratio_scores(ones=441, zeros=59)  # mean 0.882, n=500
    entry = find(MeanAndCI()(scores))
    ci = entry.metadata['ci']

    assert ci['low'] == pytest.approx(0.8510, abs=0.002)
    assert ci['high'] == pytest.approx(0.9074, abs=0.002)
    assert ci['method'] == 'wilson'
    assert ci['level'] == 0.95


def test_single_entry_with_mean_aggregation_name():
    entries = MeanAndCI()(ratio_scores(ones=441, zeros=59))

    assert len(entries) == 1
    entry = entries[0]
    # downstream consumers looking up aggregation == 'mean' keep working unchanged
    assert entry.aggregation == 'mean'
    assert entry.score == 0.882
    assert entry.num == 500
    assert entry.num_correct == 441
    assert entry.metadata['ci']['low'] is not None


def test_no_duplicate_metric_identity():
    scores = ratio_scores(ones=8, zeros=2)
    entries = MeanAndCI()(scores)

    assert len(entries) == 1
    identities = {(e.metric_name, e.aggregation) for e in entries}
    assert len(identities) == 1


def test_continuous_metric_uses_bootstrap_and_is_deterministic():
    values = [0.5 + 0.01 * (i % 50) for i in range(300)]
    first = find(MeanAndCI()(continuous_scores(values)), metric='judge_score')
    second = find(MeanAndCI()(continuous_scores(values)), metric='judge_score')
    ci = first.metadata['ci']

    assert ci['method'] == 'bootstrap'
    assert ci['low'] == second.metadata['ci']['low']
    assert ci['high'] == second.metadata['ci']['high']
    assert ci['low'] < first.score < ci['high']


def test_small_ratio_subset_flags_both_conditions():
    scores = ratio_scores(ones=17, zeros=3)  # mean 0.85, n=20
    ci = find(MeanAndCI()(scores)).metadata['ci']

    flags = ci['flags']
    assert 'low_confidence' in flags
    assert 'wide_interval' in flags


def test_large_narrow_subset_has_no_flags():
    scores = ratio_scores(ones=441, zeros=59)  # width ~5.6pt, n=500
    assert find(MeanAndCI()(scores)).metadata['ci']['flags'] == []


def test_repeated_generations_are_rejected():
    scores = ratio_scores(ones=4, zeros=1)
    duplicated = scores + [SampleScore(sample_id='s-0', score=Score(value={'accuracy': 0.0}))]

    with pytest.raises(ValueError, match='not independent samples'):
        MeanAndCI()(duplicated)


def test_empty_input_returns_empty_list():
    assert MeanAndCI()([]) == []


def test_matches_mean_aggregator_exactly():
    scores = ratio_scores(ones=7, zeros=3)
    mean_entry = Mean()(scores)[0]
    ci_entry = find(MeanAndCI()(scores))

    assert ci_entry.score == mean_entry.score
    assert ci_entry.num == mean_entry.num
    assert ci_entry.num_correct == mean_entry.num_correct


def test_report_renderer_does_not_leak_intervals_across_subsets():
    """Issue review reproduction: two accuracy subsets (90%/10%) must show their own
    intervals — the v1 cross-subset map kept the first interval for both rows."""
    from types import SimpleNamespace

    from evalscope.api.metric import MetricSelector
    from evalscope.report import gen_table
    from evalscope.report.generator import ReportGenerator

    subset_scores = {
        'easy': MeanAndCI()([SampleScore(score=Score(value={'accuracy': 1.0}), sample_id=f'easy_{i}') for i in range(100)]),
        'hard': MeanAndCI()([SampleScore(score=Score(value={'accuracy': 0.0}), sample_id=f'hard_{i}') for i in range(100)]),
    }
    adapter = SimpleNamespace(
        name='mmmu',
        pretty_name='MMMU',
        description='',
        category_map={},
        primary_metric=MetricSelector(name='accuracy'),
    )
    report = ReportGenerator.generate_report(subset_scores, 'model', adapter)

    table = gen_table(report_list=[report])
    lines = [line for line in table.splitlines() if 'easy' in line or 'hard' in line]
    assert len(lines) == 2
    assert '100%' in lines[0] and '0%' in lines[1]
    # each row carries its own interval, not the first row's
    # each row carries its own Wilson interval through the metric's semantics
    # (n=100 all-correct: [96.3%, 100%]; n=100 all-wrong: [0%, 3.6%])
    assert '[96.3%, 100%]' in lines[0]
    assert '[0%, 3.7%]' in lines[1]
