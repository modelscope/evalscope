import pytest
from pydantic import ValidationError

from evalscope.api.metric.semantics import MetricIdentity, MetricSelector
from evalscope.metrics.semantics.identity import migrate_legacy_identity


def test_identity_sorts_dimensions_and_builds_stable_key() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'target': 'answer', 'level': 'overall'})
    assert list(identity.dimensions) == ['level', 'target']
    assert identity.key == 'accuracy:mean[level=overall,target=answer]'


def test_frozen_identity_dimensions_cannot_mutate() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'target': 'answer'})

    with pytest.raises(TypeError, match='immutable'):
        identity.dimensions['target'] = 'figure'


@pytest.mark.parametrize('field,value', [('name', 'F1'), ('name', 'pass@1'), ('aggregation', 'Macro Mean')])
def test_identity_rejects_non_canonical_names(field: str, value: str) -> None:
    values = {'name': 'accuracy', 'aggregation': 'mean'}
    values[field] = value
    with pytest.raises(ValidationError):
        MetricIdentity(**values)


def test_selector_dimensions_are_partial_constraints() -> None:
    selector = MetricSelector(name='accuracy', aggregation='mean', dimensions={'target': 'answer'})
    identity = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'level': 'overall', 'target': 'answer'})
    assert selector.matches(identity)


@pytest.mark.parametrize(
    ('legacy_name', 'aggregation', 'expected'),
    [
        ('mean_acc', 'identity', ('accuracy', 'mean', {})),
        ('Bleu_4', 'mean', ('bleu', 'mean', {
            'ngram': 4
        })),
        ('ACC@0.5', 'mean', ('accuracy', 'mean', {
            'threshold': 0.5
        })),
        ('all/success_rate', 'avg@8', ('success_rate', 'mean', {
            'k': 8,
            'scope': 'all'
        })),
        ('acc_pass@16', 'mean', ('accuracy', 'pass_at_k', {
            'k': 16
        })),
        ('Act.EM', 'mean', ('exact_match', 'mean', {
            'target': 'action'
        })),
        ('mean_total_wall_time_s', 'identity', ('total_wall_time', 'mean', {})),
    ],
)
def test_legacy_names_migrate_to_structured_identity(
    legacy_name: str,
    aggregation: str,
    expected: tuple,
) -> None:
    identity = migrate_legacy_identity(legacy_name, aggregation)
    assert (identity.name, identity.aggregation, identity.dimensions) == expected


def test_hallusion_dynamic_prefix_becomes_dimensions() -> None:
    identity = migrate_legacy_identity('Overall_aAcc', 'f1', benchmark_name='hallusion_bench')
    assert identity == MetricIdentity(
        name='accuracy', aggregation='mean', dimensions={
            'level': 'overall',
            'target': 'answer'
        }
    )


@pytest.mark.parametrize(
    ('benchmark', 'legacy_name', 'expected'),
    [
        (
            'longmemeval',
            'overall_acc',
            MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'overall'}),
        ),
        (
            'locomo',
            'task_averaged_f1',
            MetricIdentity(name='f1', aggregation='macro_mean', dimensions={'scope': 'question_types'}),
        ),
        (
            'openai_mrcr',
            '8000-16000_mrcr_score',
            MetricIdentity(name='mrcr_score', aggregation='mean', dimensions={
                'min_tokens': 8000,
                'max_tokens': 16000
            }),
        ),
        (
            'wide_search',
            'pass@4_all/success_rate',
            MetricIdentity(name='success_rate', aggregation='pass_at_k', dimensions={
                'k': 4,
                'scope': 'all'
            }),
        ),
    ],
)
def test_known_dynamic_benchmark_names_migrate_without_guessing(
    benchmark: str, legacy_name: str, expected: MetricIdentity
) -> None:
    assert migrate_legacy_identity(legacy_name, 'identity', benchmark_name=benchmark) == expected
