"""The priority order that resolves a metric identity into semantics.

``SemanticsResolver.resolve`` is the only place this order is expressed, and reading a v1 report is
the same call with the stored spelling passed in. Before that, ``hydrate_report_semantics`` carried
its own chain that consulted the read-old manifest *before* the benchmark override, so the same
identity could mean one thing in an archived report and another in a fresh run.
"""
from typing import Optional

import pytest

from evalscope.api.metric.semantics import MetricIdentity, MetricKind
from evalscope.metrics.semantics.catalog import LEGACY_METRIC_MIGRATIONS, METRIC_DEFINITIONS
from evalscope.metrics.semantics.resolver import SemanticsSource, get_semantics_resolver
from evalscope.report.report import Report


def _v1_report(dataset_name: str, metric_name: str, score: float = 0.5) -> Report:
    return Report.from_dict({
        'dataset_name': dataset_name,
        'metrics': [{
            'name': metric_name,
            'score': score,
            'categories': [],
        }],
    })


@pytest.mark.parametrize('canonical_name', sorted(METRIC_DEFINITIONS))
def test_reading_a_canonical_name_matches_resolving_it_fresh(canonical_name: str) -> None:
    """A canonical name must mean the same thing however the report was written.

    The read-old manifest may only override this when it declares the name explicitly, which is
    reserved for a spelling whose historical meaning genuinely differs.
    """
    identity = MetricIdentity(name=canonical_name, aggregation='identity')
    fresh = get_semantics_resolver().resolve('equivalence_probe', identity)
    migrated = get_semantics_resolver().resolve('equivalence_probe', identity, canonical_name)

    if canonical_name in LEGACY_METRIC_MIGRATIONS:
        assert migrated.source is SemanticsSource.LEGACY_MANIFEST
        return
    assert migrated.semantics == fresh.semantics
    assert migrated.source is fresh.source


def test_only_declared_names_may_read_differently_than_they_resolve() -> None:
    """Pins the exception list, so adding one is a deliberate act rather than a side effect."""
    diverging = sorted(
        name for name, entry in LEGACY_METRIC_MIGRATIONS.items()
        if name in METRIC_DEFINITIONS and METRIC_DEFINITIONS[name] != entry
    )

    assert diverging == ['error_rate']


class TestBenchmarkOverrideWinsOverReadOld:
    """A benchmark override rejects an ambiguous name; a historical spelling cannot revive it."""

    def test_v1_report_honours_the_override(self) -> None:
        report = _v1_report('job_bench', 'total_score', 7.0)
        metric = report.metrics[0]

        assert metric.identity == MetricIdentity(name='judge_score', aggregation='identity')
        assert metric.score == 7.0
        assert metric.semantics.semantic_id == 'diagnostic.unspecified'
        assert metric.semantics.kind is MetricKind.DIAGNOSTIC
        assert report.primary_metric_identity is None

    def test_fresh_resolution_agrees_with_the_v1_report(self) -> None:
        identity = MetricIdentity(name='judge_score', aggregation='identity')
        resolved = get_semantics_resolver().resolve('job_bench', identity)

        assert resolved.source is SemanticsSource.BENCHMARK_OVERRIDE
        assert resolved.semantics.kind is MetricKind.DIAGNOSTIC

    def test_the_same_name_keeps_its_quality_semantics_elsewhere(self) -> None:
        """The override is scoped to one benchmark, so it must not leak into others."""
        resolved = get_semantics_resolver().resolve('mia_bench', MetricIdentity(name='judge_score',
                                                                               aggregation='mean'))

        assert resolved.semantics.semantic_id == 'quality.judge_score.unbounded'
        assert resolved.semantics.kind is MetricKind.QUALITY


class TestReadOldWinsOverTheCanonicalTable:
    """``error_rate`` graded outcomes now; a v1 report stored a parse-status share under that name."""

    def test_v1_report_keeps_the_historical_meaning(self) -> None:
        metric = _v1_report('legacy_parser', 'error_rate', 0.2).metrics[0]

        assert metric.semantics.semantic_id == 'diagnostic.parse_status.ratio'
        assert metric.semantics.kind is MetricKind.DIAGNOSTIC

    def test_fresh_result_uses_the_current_meaning(self) -> None:
        resolved = get_semantics_resolver().resolve('current', MetricIdentity(name='error_rate', aggregation='mean'))

        assert resolved.semantics.semantic_id == 'quality.error_rate.ratio'
        assert resolved.semantics.kind is MetricKind.QUALITY


@pytest.mark.parametrize(
    ('identity', 'legacy_name', 'expected_source'),
    [
        (MetricIdentity(name='judge_score', aggregation='identity'), None, SemanticsSource.BENCHMARK_OVERRIDE),
        (MetricIdentity(name='judge_score', aggregation='identity'), 'total_score',
         SemanticsSource.BENCHMARK_OVERRIDE),
        (MetricIdentity(name='accuracy', aggregation='mean'), 'mean_acc', SemanticsSource.LEGACY_MANIFEST),
        (MetricIdentity(name='accuracy', aggregation='pass_at_k', dimensions={'k': 4}), None,
         SemanticsSource.AGGREGATION_OVERRIDE),
        (MetricIdentity(name='accuracy', aggregation='mean'), None, SemanticsSource.METRIC_NAME),
        (MetricIdentity(name='vendor_metric', aggregation='mean'), None, SemanticsSource.DIAGNOSTIC_FALLBACK),
    ],
)
def test_each_step_of_the_order_is_reachable_and_reported(
    identity: MetricIdentity, legacy_name: Optional[str], expected_source: SemanticsSource
) -> None:
    resolved = get_semantics_resolver().resolve('job_bench', identity, legacy_name)

    assert resolved.source is expected_source
