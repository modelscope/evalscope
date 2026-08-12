import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.metric.semantics import MetricIdentity
from evalscope.metrics.aggregators.aggregators import MeanPassAtK, MeanPassHatK, MeanVoteAtK


def _scores(values, predictions=None):
    predictions = predictions or [None] * len(values)
    return [
        SampleScore(
            sample_id=index,
            group_id=f'group-{index // 2}',
            score=Score(value={'accuracy': value}, extracted_prediction=prediction),
        )
        for index, (value, prediction) in enumerate(zip(values, predictions))
    ]


@pytest.mark.parametrize(
    ('aggregator', 'values', 'predictions', 'aggregation', 'expected'),
    [
        (MeanPassAtK(), [1, 0, 1, 1], None, 'pass_at_k', [0.75, 1.0]),
        (MeanPassHatK(), [1, 0, 1, 1], None, 'pass_hat_k', [0.75, 0.5]),
        (MeanVoteAtK(), [1, 0, 0, 1], ['a', 'b', 'c', 'd'], 'vote_at_k', [0.5, 0.5]),
    ],
)
def test_k_aggregators_emit_structured_identities_without_mutating_samples(
    aggregator, values, predictions, aggregation: str, expected
) -> None:
    scores = _scores(values, predictions)

    aggregates = aggregator(scores)

    structured = [aggregate for aggregate in aggregates if aggregate.aggregation == aggregation]
    assert [aggregate.identity for aggregate in structured] == [
        MetricIdentity(name='accuracy', aggregation=aggregation, dimensions={'k': 1}),
        MetricIdentity(name='accuracy', aggregation=aggregation, dimensions={'k': 2}),
    ]
    assert [aggregate.score for aggregate in structured] == pytest.approx(expected)
    assert all(list(sample.score.value) == ['accuracy'] for sample in scores)
