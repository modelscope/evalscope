"""Metric audit: a read-only report of which final report metric names EvalScope can emit and
whether the metric semantics catalog covers them.

Entry point::

    python -m evalscope.metrics.semantics.audit [--json] [--benchmark NAME ...] [--observed-path PATH ...]

or ``make metric-audit``. Exits non-zero when an audit error is found; the caller decides whether
that blocks a pipeline.

The two lists that drive the catalog completion work are ``E_UNDECLARED_METRIC`` (add a metric
name entry) and ``E_PRIMARY_COUNT`` (declare ``BenchmarkMeta.primary_metric``): both name the
benchmark, the metric and the exact location, so neither has to be enumerated by hand.

This subpackage is a maintenance entry point and is deliberately not re-exported by
``evalscope.metrics.semantics``, so a report read never pulls in the AST scanning machinery.
"""

from .checks import AuditError, AuditErrorCode, AuditReport, run_checks
from .cli import format_audit_report, main, run_audit
from .collectors import (
    MetricGroup,
    collect_custom_aggregation_names,
    collect_declared_metrics,
    collect_default_aggregation_names,
    collect_metric_inventory,
    collect_observed_metrics,
    collect_perf_field_keys,
)

__all__ = [
    # entry points
    'main',
    'run_audit',
    'format_audit_report',
    # collectors
    'collect_custom_aggregation_names',
    'collect_declared_metrics',
    'collect_default_aggregation_names',
    'collect_metric_inventory',
    'collect_observed_metrics',
    'collect_perf_field_keys',
    # findings
    'AuditError',
    'AuditErrorCode',
    'AuditReport',
    'MetricGroup',
    'run_checks',
]
