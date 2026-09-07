import pytest

from evalscope.api.metric.semantics import MetricIdentity, MetricKind
from evalscope.metrics.semantics.resolver import SemanticsSource, get_semantics_resolver


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
