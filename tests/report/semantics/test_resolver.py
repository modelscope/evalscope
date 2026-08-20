import pytest

from evalscope.api.metric.semantics import MetricIdentity, MetricKind, MetricSelector
from evalscope.metrics.semantics.resolver import SemanticsSource, get_semantics_resolver, select_primary_identity


def test_resolver_uses_canonical_name() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean')
    resolved = get_semantics_resolver().resolve('gsm8k', identity)

    assert resolved.source is SemanticsSource.METRIC_NAME
    assert resolved.semantics.semantic_id == 'quality.accuracy.ratio'
    assert resolved.semantics.kind is MetricKind.QUALITY


def test_unknown_metric_degrades_to_diagnostic() -> None:
    identity = MetricIdentity(name='third_party_measure', aggregation='mean')
    resolved = get_semantics_resolver().resolve('third_party', identity)

    assert resolved.degraded
    assert resolved.semantics.kind is MetricKind.DIAGNOSTIC


@pytest.mark.parametrize(
    ('name', 'expected_display_name'),
    [('is_incorrect', 'Incorrect rate'), ('is_not_attempted', 'Not attempted rate')],
)
def test_three_way_grading_diagnostics_have_explicit_display_names(name: str, expected_display_name: str) -> None:
    identity = MetricIdentity(name=name, aggregation='mean')
    resolved = get_semantics_resolver().resolve('chinese_simpleqa', identity)

    assert resolved.semantics.kind is MetricKind.DIAGNOSTIC
    assert resolved.semantics.display_name == expected_display_name


def test_pass_at_k_dimensions_share_one_aggregation_semantics() -> None:
    resolver = get_semantics_resolver()
    for k in (1, 7, 137):
        identity = MetricIdentity(name='accuracy', aggregation='pass_at_k', dimensions={'k': k})
        assert resolver.resolve('humaneval', identity).semantics.semantic_id == 'quality.pass_at_k.ratio'


def test_structured_selector_selects_one_identity_without_mutating_semantics() -> None:
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
    semantics = {identity.key: resolver.resolve('general_qa', identity).semantics for identity in identities}
    selector = MetricSelector(
        name='rouge', aggregation='mean', dimensions={
            'variant': 'l',
            'statistic': 'recall'
        }
    )

    primary = select_primary_identity(identities, semantics, selector)

    assert primary == identities[1]
    assert all(item.kind is MetricKind.QUALITY for item in semantics.values())


@pytest.mark.parametrize('selector', [MetricSelector(name='recall'), MetricSelector(name='accuracy')])
def test_selector_zero_or_multiple_matches_fails(selector: MetricSelector) -> None:
    identities = [
        MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'a'}),
        MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'b'}),
    ]
    resolver = get_semantics_resolver()
    semantics = {identity.key: resolver.resolve('benchmark', identity).semantics for identity in identities}

    with pytest.raises(ValueError, match='matched'):
        select_primary_identity(identities, semantics, selector)


def test_only_one_quality_identity_can_be_implicit_primary() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean')
    semantics = get_semantics_resolver().resolve('benchmark', identity).semantics

    assert select_primary_identity([identity], {identity.key: semantics}, None) == identity
