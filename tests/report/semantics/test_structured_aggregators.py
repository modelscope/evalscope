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
    assert [aggregate.num for aggregate in structured] == [2, 2]
    assert all(list(sample.score.value) == ['accuracy'] for sample in scores)


def test_pass_at_k_averages_unique_groups_instead_of_weighting_repetitions() -> None:
    scores = [
        SampleScore(sample_id=0, group_id='short', score=Score(value={'accuracy': 1})),
        SampleScore(sample_id=1, group_id='long', score=Score(value={'accuracy': 0})),
        SampleScore(sample_id=2, group_id='long', score=Score(value={'accuracy': 0})),
    ]

    aggregates = MeanPassAtK()(scores)
    pass_at_1 = next(aggregate for aggregate in aggregates if aggregate.aggregation == 'pass_at_k')

    assert pass_at_1.score == pytest.approx(0.5)
    assert pass_at_1.num == 2
    assert pass_at_1.ids == ['short', 'long']
