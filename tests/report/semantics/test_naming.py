import pytest
from pydantic import ValidationError

from evalscope.api.metric.semantics import MetricIdentity, MetricSelector
from evalscope.metrics.semantics.catalog import LEGACY_METRIC_MIGRATIONS
from evalscope.metrics.semantics.identity import migrate_legacy_identity
from evalscope.metrics.semantics.legacy import LEGACY_METRIC_ALIASES


def test_identity_sorts_dimensions_and_builds_stable_key() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'target': 'answer', 'level': 'overall'})
    assert list(identity.dimensions) == ['level', 'target']
    assert identity.key == 'accuracy:mean[level="overall",target="answer"]'


def test_identity_key_preserves_dimension_types_and_boundaries() -> None:
    numeric = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'k': 1})
    text = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'k': '1'})
    embedded_delimiters = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'a': 'x,b=y'})
    separate_dimensions = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'a': 'x', 'b': 'y'})

    assert numeric.key != text.key
    assert embedded_delimiters.key != separate_dimensions.key


def test_identity_comparison_keeps_booleans_distinct_and_normalizes_json_numbers() -> None:
    boolean = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'value': True})
    integer = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'value': 1})
    integral_float = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'value': 1.0})
    negative_zero = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'value': -0.0})

    assert boolean != integer
    assert len({boolean, integer}) == 2
    assert integer == integral_float
    assert integer.key == integral_float.key == 'accuracy:mean[value=1]'
    assert negative_zero.key == 'accuracy:mean[value=0]'


def test_selector_does_not_match_boolean_to_numeric_dimension() -> None:
    selector = MetricSelector(name='accuracy', dimensions={'value': True})
    identity = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'value': 1})

    assert not selector.matches(identity)


def test_frozen_identity_rejects_field_assignment() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'target': 'answer'})

    with pytest.raises(ValidationError):
        identity.dimensions = {'target': 'figure'}


def test_identity_equality_and_hash_ignore_dimension_order() -> None:
    # `dimensions` is a plain dict; what makes an identity stable is that equality and hash are
    # derived from the normalized `sort_key`, not from the mapping's insertion order.
    left = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'target': 'answer', 'level': 'overall'})
    right = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'level': 'overall', 'target': 'answer'})

    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


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
        ('bleu-4', 'mean', ('bleu', 'mean', {
            'ngram': 4
        })),
        ('Rouge-L-R', 'mean', ('rouge', 'mean', {
            'statistic': 'recall',
            'variant': 'l'
        })),
        ('Rouge-2-F', 'mean', ('rouge', 'mean', {
            'ngram': 2,
            'statistic': 'f1'
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


def test_exact_alias_manifest_drives_identity_and_read_old_semantics() -> None:
    for name, alias in LEGACY_METRIC_ALIASES.items():
        assert migrate_legacy_identity(name, 'identity').name == alias.canonical_name
        assert (name in LEGACY_METRIC_MIGRATIONS) is (alias.baseline is not None)
