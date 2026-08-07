"""Final report metric name composition.

The *final report metric name* is the string that ``ReportGenerator.generate_report()``
writes into ``Metric.name``. It is the key the semantics catalog and the metric audit are keyed
by, so the spelling rule must live in exactly one place.

Any consumer that needs to know the final report metric name of an ``AggScore``
(report generation, catalog lookups, audits) MUST use
:func:`compose_final_metric_name` instead of re-implementing the rule.
"""

from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from evalscope.api.metric import AggScore

__all__ = ['compose_final_metric_name', 'match_primary_final_name']


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


def match_primary_final_name(
    primary_metric: Optional[str],
    metric_names: Sequence[str],
    aggregation: Optional[str] = None,
) -> Optional[str]:
    """Map a raw ``BenchmarkMeta.primary_metric`` onto the final report metric name.

    ``primary_metric`` names a raw ``metric_list`` entry while a report stores the *final* name,
    which may carry the aggregation prefix. Candidates are tried from the most explicit to the
    least, and an ambiguous match yields ``None`` rather than an arbitrary pick:

    1. the raw name itself, for benchmarks whose aggregation adds no prefix
    2. ``f'{aggregation}_{primary_metric}'``, the spelling of :func:`compose_final_metric_name`
    3. the unique remaining name whose prefix-stripped remainder starts with the raw name, which
       covers a metric that renames itself while aggregating (``pass`` -> ``mean_pass_rate``)

    Args:
        primary_metric: Raw metric name declared as primary, or ``None``.
        metric_names: Final report metric names present in the report.
        aggregation: ``BenchmarkMeta.aggregation`` of the benchmark, when known.

    Returns:
        The matching final report metric name, or ``None`` when nothing matches unambiguously.
    """
    if not primary_metric:
        return None

    if primary_metric in metric_names:
        return primary_metric

    if aggregation:
        prefixed = f'{aggregation}_{primary_metric}'
        if prefixed in metric_names:
            return prefixed

    prefix = f'{aggregation}_' if aggregation else ''
    candidates = [
        name for name in metric_names
        if name.startswith(prefix) and name[len(prefix):].startswith(f'{primary_metric}_')
    ]
    return candidates[0] if len(candidates) == 1 else None
