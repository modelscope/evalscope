"""Metric semantics data layer.

The contract models live in ``evalscope.api.metric.semantics``; this package holds the data and
the logic built on it: the common baseline table, the central catalog (three tables keyed by
final report metric name), the resolver, the primary metric summary helper and the value
formatting primitives.

This module is the public surface of the package, so consumers do not have to know which
concrete module owns which symbol::

    from evalscope.metrics.semantics import format_metric_value, get_semantics_resolver

    resolved = get_semantics_resolver().resolve('gsm8k', 'AverageAccuracy')
    text = format_metric_value(0.857, resolved.semantics)

Importing this package pulls in the whole catalog and validates it, which is why read paths
inside ``evalscope.report`` import these names through a function-local lazy import: ``report.py``
stays importable on its own and no ``report`` <-> ``metrics.semantics`` import cycle can form.
Nothing in this package imports ``evalscope.report`` at module level.

``audit/`` is deliberately not re-exported: it is a maintenance entry point, not part of the
runtime API, and importing it here would drag the audit dependencies into every report read.

Only the two formatters the backend actually renders with are exported. ``formatting.py`` also
holds the ``FormattedMetric`` mirror of the frontend primitive, which exists so the two sides can
be asserted against the same golden samples; it is imported from that module directly rather than
re-exported, so the package surface stays limited to what production calls.
"""

from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.catalog import (
    BENCHMARK_DYNAMIC_METRICS,
    BENCHMARK_METRIC_OVERRIDES,
    METRIC_NAME_SEMANTICS,
    lookup_metric_entry,
)
from evalscope.metrics.semantics.formatting import format_metric_value, format_raw_metric_value
from evalscope.metrics.semantics.naming import compose_final_metric_name
from evalscope.metrics.semantics.resolver import (
    AUDIT_MESSAGE_PREFIX,
    DIAGNOSTIC_FALLBACK_SEMANTIC_ID,
    ResolvedSemantics,
    SemanticsResolver,
    SemanticsSource,
    UndeclaredMetricError,
    builtin_benchmark_names,
    catalog_entry_location,
    diagnostic_fallback,
    get_semantics_resolver,
    hydrate_report_semantics,
    is_builtin_benchmark,
    is_public_perf_field,
)
from evalscope.metrics.semantics.summary import (
    MetricSummary,
    PrimaryMetricRef,
    SummaryStatus,
    summarize_primary_metrics,
)

__all__ = [
    # catalog data
    'BENCHMARK_DYNAMIC_METRICS',
    'BENCHMARK_METRIC_OVERRIDES',
    'METRIC_NAME_SEMANTICS',
    'SEMANTIC_BASELINES',
    'lookup_metric_entry',
    # naming
    'compose_final_metric_name',
    # resolution
    'AUDIT_MESSAGE_PREFIX',
    'DIAGNOSTIC_FALLBACK_SEMANTIC_ID',
    'ResolvedSemantics',
    'SemanticsResolver',
    'SemanticsSource',
    'UndeclaredMetricError',
    'builtin_benchmark_names',
    'catalog_entry_location',
    'diagnostic_fallback',
    'get_semantics_resolver',
    'hydrate_report_semantics',
    'is_builtin_benchmark',
    'is_public_perf_field',
    # summary
    'MetricSummary',
    'PrimaryMetricRef',
    'SummaryStatus',
    'summarize_primary_metrics',
    # formatting
    'format_metric_value',
    'format_raw_metric_value',
]
