"""Final report metric name composition.

The *final report metric name* is the string that ``ReportGenerator.generate_report()``
writes into ``Metric.name``. It is the key used by the semantics catalog, the legacy
mapping table and the audit script, so the spelling rule must live in exactly one place.

Any consumer that needs to know the final report metric name of an ``AggScore``
(report generation, catalog lookups, audits) MUST use
:func:`compose_final_metric_name` instead of re-implementing the rule.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalscope.api.metric import AggScore

__all__ = ['compose_final_metric_name']


def compose_final_metric_name(agg_score: 'AggScore', add_aggregation_name: bool = True) -> str:
    """Compose the final report metric name of an aggregated score.

    Mirrors the spelling rule applied by ``ReportGenerator.generate_report()``:
    when ``add_aggregation_name`` is enabled and the score carries a non-empty
    ``aggregation_name``, the aggregation name is prefixed to the metric name.

    Args:
        agg_score: Aggregated score whose ``aggregation_name`` and ``metric_name`` are used.
        add_aggregation_name: Whether the aggregation name may be prefixed, matching
            ``DataAdapter.add_aggregation_name``.

    Returns:
        str: ``f'{aggregation_name}_{metric_name}'`` when the aggregation name is used,
        otherwise ``metric_name``.
    """
    if add_aggregation_name and agg_score.aggregation_name:
        return f'{agg_score.aggregation_name}_{agg_score.metric_name}'
    return agg_score.metric_name
