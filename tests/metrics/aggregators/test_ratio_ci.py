"""Interval estimates for ratio metrics (#2011, first slice)."""

import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.metrics.aggregators import Mean
from evalscope.metrics.aggregators.ci import (
    DEFAULT_CI_LEVEL,
    MAX_CONFIDENT_CI_WIDTH,
    MIN_CONFIDENT_SAMPLES,
    ratio_ci_metadata,
    wilson_interval,
)


def _scores(values: dict) -> list:
    return [SampleScore(sample_id=i, score=Score(value={name: value for name, value in values.items()}))
            for i, value in enumerate([0])]


def test_wilson_matches_known_values() -> None:
    """p=0.5, n=100, 95% -> [0.4038, 0.5962]."""
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_stays_inside_the_unit_interval_at_the_edges() -> None:
    """The normal approximation collapses at p=0/p=1; Wilson keeps a non-zero width."""
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert 0.2 < high < 0.3

    low, high = wilson_interval(10, 10)
    assert high == 1.0
    assert 0.7 < low < 0.8


def test_wilson_without_samples_is_degenerate() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_metadata_describes_the_interval() -> None:
    metadata = ratio_ci_metadata(8, 10)

    assert metadata['ci_method'] == 'wilson'
    assert metadata['ci_level'] == DEFAULT_CI_LEVEL
    assert metadata['ci_low'] < 0.8 < metadata['ci_high']
    assert metadata['low_confidence'] is True  # 10 samples is below the threshold


def test_metadata_marks_small_samples_and_wide_intervals_as_low_confidence() -> None:
    assert ratio_ci_metadata(15, MIN_CONFIDENT_SAMPLES - 1)['low_confidence'] is True
    assert ratio_ci_metadata(95, 100)['low_confidence'] is False

    wide = ratio_ci_metadata(5, 10)
    assert (wide['ci_high'] - wide['ci_low']) > MAX_CONFIDENT_CI_WIDTH
    assert wide['low_confidence'] is True


def test_mean_attaches_interval_to_ratio_metrics() -> None:
    scores = [SampleScore(sample_id=i, score=Score(value={'acc': value})) for i, value in enumerate([1, 0, 1, 1])]
    result = Mean()(scores)

    assert len(result) == 1
    aggregate = result[0]
    assert aggregate.num_correct == 3
    assert aggregate.metadata is not None
    assert aggregate.metadata['ci_method'] == 'wilson'
    assert aggregate.metadata['ci_low'] < aggregate.score < aggregate.metadata['ci_high']


def test_mean_leaves_continuous_metrics_without_an_interval() -> None:
    scores = [SampleScore(sample_id=i, score=Score(value={'bleu': value})) for i, value in enumerate([0.4, 0.6])]
    result = Mean()(scores)

    assert result[0].num_correct is None
    assert result[0].metadata is None
