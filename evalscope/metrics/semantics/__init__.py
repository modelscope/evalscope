"""Metric semantics data layer.

The contract models live in ``evalscope.api.metric.semantics``; this package holds the data and
the logic built on it: the common baseline table, the central catalog (two tables keyed by final
report metric name), the resolver, the primary metric summary helper and the value formatting
primitives.

This module is the public surface of the package, so consumers do not have to know which
concrete module owns which symbol::

    from evalscope.metrics.semantics import format_metric_value, get_semantics_resolver

    resolved = get_semantics_resolver().resolve('gsm8k', 'AverageAccuracy')
    text = format_metric_value(0.857, resolved.semantics)

Importing this package pulls in the whole catalog and validates it, which is why read paths
inside ``evalscope.report`` import these names through a function-local lazy import: ``report.py``
stays importable on its own and no ``report`` <-> ``metrics.semantics`` import cycle can form.
Nothing in this package imports ``evalscope.report`` at module level.

The surface is exactly what production imports. Everything else -- the catalog tables, the
baseline table, the resolver internals and private formatting helpers -- stays in its owning
module, so tests import those directly from ``catalog.py`` / ``baselines.py`` / ``resolver.py`` /
``formatting.py``.
"""

from evalscope.metrics.semantics.formatting import format_metric_label, format_metric_labels, format_metric_value
from evalscope.metrics.semantics.resolver import get_semantics_resolver, hydrate_report_semantics
from evalscope.metrics.semantics.summary import PrimaryMetricRef

__all__ = [
    # resolution
    'get_semantics_resolver',
    'hydrate_report_semantics',
    # primary metric references
    'PrimaryMetricRef',
    # formatting
    'format_metric_label',
    'format_metric_labels',
    'format_metric_value',
]
