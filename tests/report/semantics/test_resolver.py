import pytest

from evalscope.api.metric.semantics import MetricIdentity, MetricRole, MetricSelector
from evalscope.metrics.semantics.resolver import SemanticsSource, attribute_metric_roles, get_semantics_resolver


def test_resolver_uses_canonical_name_without_alias_lookup() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean')
    resolved = get_semantics_resolver().resolve('gsm8k', identity)
    assert resolved.source is SemanticsSource.METRIC_NAME
    assert resolved.semantics.semantic_id == 'quality.accuracy.ratio'
    assert resolved.semantics.role is MetricRole.AUXILIARY


def test_unknown_canonical_metric_degrades_to_diagnostic() -> None:
    identity = MetricIdentity(name='third_party_measure', aggregation='mean')
    resolved = get_semantics_resolver().resolve('third_party', identity)
    assert resolved.degraded
    assert resolved.semantics.role is MetricRole.DIAGNOSTIC


def test_any_pass_at_k_dimension_uses_one_aggregation_override() -> None:
    resolver = get_semantics_resolver()
    for k in (1, 7, 137):
        identity = MetricIdentity(name='accuracy', aggregation='pass_at_k', dimensions={'k': k})
        assert resolver.resolve('humaneval', identity).semantics.semantic_id == 'quality.pass_at_k.ratio'


def test_explicit_selector_assigns_primary_once() -> None:
    identities = [
        MetricIdentity(name='accuracy', aggregation='mean'),
        MetricIdentity(name='f1', aggregation='mean'),
    ]
    resolver = get_semantics_resolver()
    base = {identity.key: resolver.resolve('benchmark', identity).semantics for identity in identities}
    attributed, primary = attribute_metric_roles(identities, base, MetricSelector(name='f1'))
    assert primary == identities[1]
    assert attributed[identities[0].key].role is MetricRole.AUXILIARY
    assert attributed[identities[1].key].role is MetricRole.PRIMARY


def test_structured_selector_disambiguates_rouge_variants() -> None:
    identities = [
        MetricIdentity(name='rouge', aggregation='mean', dimensions={
            'ngram': 1,
            'statistic': 'recall'
        }),
        MetricIdentity(name='rouge', aggregation='mean', dimensions={
            'statistic': 'recall',
            'variant': 'l'
        }),
    ]
    resolver = get_semantics_resolver()
    base = {identity.key: resolver.resolve('general_qa', identity).semantics for identity in identities}
    selector = MetricSelector(
        name='rouge', aggregation='mean', dimensions={
            'variant': 'l',
            'statistic': 'recall'
        }
    )

    _, primary = attribute_metric_roles(identities, base, selector)

    assert primary == identities[1]


@pytest.mark.parametrize('selector', [MetricSelector(name='recall'), MetricSelector(name='accuracy')])
def test_selector_zero_or_multiple_matches_fails(selector: MetricSelector) -> None:
    identities = [
        MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'a'}),
        MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'b'}),
    ]
    resolver = get_semantics_resolver()
    base = {identity.key: resolver.resolve('benchmark', identity).semantics for identity in identities}
    with pytest.raises(ValueError, match='matched'):
        attribute_metric_roles(identities, base, selector)


def test_only_one_scored_identity_can_be_implicit_primary() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean')
    semantics = get_semantics_resolver().resolve('benchmark', identity).semantics
    attributed, primary = attribute_metric_roles([identity], {identity.key: semantics}, None)
    assert primary == identity
    assert attributed[identity.key].role is MetricRole.PRIMARY
