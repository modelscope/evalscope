"""AUPRC retains threshold-grouped average precision and report provenance."""

import math
import random

import pytest

from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.metric.decision import decision_score
from evalscope.api.model.decision import DecisionAnswer
from evalscope.metrics.aggregators.decision import DecisionAggregator


def _samples(probabilities: list[float], labels: list[int]) -> list[SampleScore]:
    return [
        SampleScore(
            sample_id=index,
            score=decision_score(
                DecisionAnswer(type='noul', noul=p, probabilities={'no': 1 - p, 'yes': p}),
                'yes' if label else 'no',
                ['no', 'yes'],
            ),
        )
        for index, (p, label) in enumerate(zip(probabilities, labels))
    ]


def _auprc(samples: list[SampleScore]) -> AggScore:
    return next(metric for metric in DecisionAggregator()(samples) if metric.metric_name == 'auprc')


def _legacy_average_precision(probabilities: list[float], labels: list[int]) -> float:
    """Frozen pre-optimization threshold scans for numerical equivalence checks."""
    positives = sum(labels)
    tp = seen = 0
    result = 0.0
    for probability in sorted(set(probabilities), reverse=True):
        indices = [index for index, value in enumerate(probabilities) if value == probability]
        added = sum(labels[index] for index in indices)
        tp += added
        seen += len(indices)
        result += added / positives * tp / seen
    return result


@pytest.mark.parametrize(
    'probabilities,labels,expected',
    [
        ([0.8], [1], 1.0),
        ([0.1, 0.4, 0.8, 0.9], [0, 1, 1, 0], 7 / 12),
        ([0.9, 0.9, 0.5, 0.2], [1, 0, 1, 0], 7 / 12),
        ([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], 0.5),
        ([0.1, 0.3, 0.9], [1, 1, 1], 1.0),
        ([0.0, 1.0], [1, 0], 0.5),
        ([0.0, 1.0], [0, 1], 1.0),
        ([-0.0, 0.0], [1, 0], 0.5),
        ([0.5, math.nextafter(0.5, 1.0)], [1, 0], 0.5),
        ([0.5, math.nextafter(0.5, 1.0)], [0, 1], 1.0),
    ],
)
def test_auprc_thresholds(probabilities: list[float], labels: list[int], expected: float) -> None:
    result = _auprc(_samples(probabilities, labels))

    assert result.score == pytest.approx(expected, abs=1e-15)
    assert result.aggregation == 'decision'
    assert result.dimensions == {'kind': 'noul'}
    assert result.num == len(labels)
    assert result.ids == list(range(len(labels)))


@pytest.mark.parametrize('seed', range(10))
@pytest.mark.parametrize('buckets', [None, 2, 10])
def test_auprc_matches_legacy_exactly(seed: int, buckets: int | None) -> None:
    rng = random.Random(seed)
    size = rng.randrange(20, 200)
    probabilities = [rng.random() if buckets is None else rng.randrange(buckets) / buckets for _ in range(size)]
    labels = [rng.randrange(2) for _ in range(size)]
    labels[0] = 1
    samples = _samples(probabilities, labels)
    # Read the scoring probabilities, including any transport normalization.
    actual_probabilities = [sample.score.metadata['decision']['probabilities'][1] for sample in samples]
    expected = _legacy_average_precision(actual_probabilities, labels)

    result = _auprc(samples)
    assert result.score == expected
    rng.shuffle(samples)
    reordered = _auprc(samples)
    assert reordered.score == expected
    assert reordered.ids == [sample.sample_id for sample in samples]


@pytest.mark.parametrize('probabilities', [[0.8], [0.0, 0.5, 1.0]])
def test_auprc_without_positives_stays_unavailable(probabilities: list[float]) -> None:
    metrics = DecisionAggregator()(_samples(probabilities, [0] * len(probabilities)))

    assert 'auprc' not in {metric.metric_name for metric in metrics}
    precision = next(metric for metric in metrics if metric.metric_name == 'precision')
    assert precision.metadata['unavailable']['auprc'] == 'No positive gold observations.'


def test_empty_decisions_have_no_metrics() -> None:
    assert DecisionAggregator()([]) == []


def test_auprc_counts_only_binary_decisions() -> None:
    binary = _samples([0.9, 0.9, 0.5, 0.2], [1, 0, 1, 0])
    choice = SampleScore(
        sample_id='choice',
        score=decision_score(
            DecisionAnswer(type='choice', choice='B', probabilities={'A': 0.1, 'B': 0.9}), 'B', ['A', 'B']
        ),
    )
    result = _auprc([choice, *binary])

    assert result.score == pytest.approx(7 / 12)
    assert result.num == 4
    assert result.ids == [0, 1, 2, 3]
    assert result.dimensions == {'kind': 'noul'}
