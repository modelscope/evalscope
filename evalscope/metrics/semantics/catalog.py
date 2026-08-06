"""Central metric semantics catalog.

The catalog answers one question only: *what does this final report metric name mean?* It is
organized by final report metric name (not by benchmark), because the 219 built-in benchmarks
produce 395 ``(benchmark, metric)`` pairs but only ~131 distinct metric names, and 149
benchmarks emit a single metric. Direction / unit / scale / precision therefore need to be
declared once per name and are reused by every benchmark.

Three tables live here:

- :data:`METRIC_NAME_SEMANTICS` -- final report metric name -> :class:`MetricEntry` (a baseline
  reference plus optional field overrides). Also holds the historical report names, grouped in a
  dedicated section so they can be dropped once no report of that vintage is opened again.
- :data:`BENCHMARK_METRIC_OVERRIDES` -- ``(benchmark_name, final_metric_name)`` -> entry, used
  *only* when the same name means different things in different benchmarks (a collision).
- :data:`BENCHMARK_DYNAMIC_METRICS` -- ``benchmark_name`` -> allow-list of final report metric
  names generated at runtime (``pass@{k}`` families, f-string composed names).

The primary metric of a benchmark is **not** declared here: it is ``BenchmarkMeta.primary_metric``
(next to ``metric_list``), applied as a role adjustment by the resolver.

Every lookup is an exact-key dictionary lookup: no regular expressions, no name normalization,
no fuzzy or magnitude based inference. Importing this module validates every entry (each
``MetricEntry`` resolves against :data:`SEMANTIC_BASELINES` and passes the contract validation),
so an illegal declaration or a dangling baseline reference aborts the import immediately.
"""

from typing import Dict, Tuple

from evalscope.api.metric.semantics import BASELINE_TABLE_LOCATION, MetricEntry
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES

#: Where to declare a metric name, used in audit and validation messages.
METRIC_NAME_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::METRIC_NAME_SEMANTICS'

#: Where to declare a benchmark level collision override, used in audit messages.
BENCHMARK_OVERRIDE_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::BENCHMARK_METRIC_OVERRIDES'

#: Where to declare a dynamic metric allow-list, used in audit messages.
DYNAMIC_METRIC_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::BENCHMARK_DYNAMIC_METRICS'

METRIC_NAME_SEMANTICS: Dict[str, MetricEntry] = {
    # --- quality ratios: one line each, reused by every benchmark ------------------------
    # Bounded [0, 1] ratios rendered as percent, higher is better.
    'mean_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'accuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    'multi_choice_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'relaxed_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'schema_accuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    # Graded-answer benchmarks (simple_qa, browsecomp, chinese_simple_qa) report the share of
    # correct answers as `is_correct`.
    'is_correct': MetricEntry(baseline='quality.accuracy.ratio'),
    # Agent style task completion ratio (miniwob, wide_search).
    'success_rate': MetricEntry(baseline='quality.accuracy.ratio'),
    # --- exact match ---------------------------------------------------------------------
    'em': MetricEntry(baseline='quality.exact_match.ratio'),
    'exact_match': MetricEntry(baseline='quality.exact_match.ratio'),
    # --- pass ratios ---------------------------------------------------------------------
    'pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'pass_at_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'pass_hat_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'Pass@1': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'pass@1': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'strict_pass': MetricEntry(baseline='quality.pass_at_k.ratio'),
    # --- F1 / precision / recall ---------------------------------------------------------
    'f1': MetricEntry(baseline='quality.f1.ratio'),
    'F1': MetricEntry(baseline='quality.f1.ratio'),
    'f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'f1_macro': MetricEntry(baseline='quality.f1.ratio'),
    'f1_micro': MetricEntry(baseline='quality.f1.ratio'),
    'f1_weighted': MetricEntry(baseline='quality.f1.ratio'),
    'simple_f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'tool_call_f1': MetricEntry(baseline='quality.f1.ratio'),
    'precision': MetricEntry(baseline='quality.precision.ratio'),
    'recall': MetricEntry(baseline='quality.recall.ratio'),
    # --- speech recognition error rates: lower is better ---------------------------------
    'wer': MetricEntry(baseline='quality.wer.ratio'),
    'mean_wer': MetricEntry(baseline='quality.wer.ratio'),
    'audio_wer': MetricEntry(baseline='quality.wer.ratio'),
    'cer': MetricEntry(baseline='quality.cer.ratio'),
    'mean_cer': MetricEntry(baseline='quality.cer.ratio'),
    # --- judge scores (unbounded) --------------------------------------------------------
    'gpt_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'total_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    # --- diagnostics: distribution shares and raw counts carry no direction --------------
    'error_rate': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'is_incorrect': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'is_not_attempted': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'yes_ratio': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'maybe_ratio': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'no_answer_num': MetricEntry(baseline='diagnostic.count.items'),
    'count_successful_tool_call': MetricEntry(baseline='diagnostic.count.items'),
    'count_finish_reason_tool_call': MetricEntry(baseline='diagnostic.count.items'),
    'count_finish_reason_tool_calls': MetricEntry(baseline='diagnostic.count.items'),
    # --- legacy names: only produced by report files written before the semantics -------
    # contract. Safe to drop once no report of that vintage is expected to be opened again.
    'AverageAccuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    'WeightedAverageAccuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    'WeightedScorePercent': MetricEntry(baseline='quality.score.points_100'),
    'AverageOutputTps': MetricEntry(baseline='perf.throughput.tokens_per_second'),
}
"""Final report metric name -> catalog entry, reused by every benchmark.

Seeded from the metric names this repository actually emits. The audit driven completion to the
full ~131 names (task 5) fills the remaining default / custom aggregation names.
"""

BENCHMARK_METRIC_OVERRIDES: Dict[Tuple[str, str], MetricEntry] = {
    # `total_score` is a judge score in mia_bench but the raw sum of passed rubric weights in
    # job_bench, i.e. an intermediate judge value: reassign the collision to a diagnostic.
    ('job_bench', 'total_score'): MetricEntry(baseline='diagnostic.unspecified'),
    ('job_bench', 'mean_total_score'): MetricEntry(baseline='diagnostic.unspecified'),
}
"""``(benchmark_name, final_metric_name)`` -> entry, only for same-name / different-meaning
collisions. Each entry carries the collision reason in a comment."""

BENCHMARK_DYNAMIC_METRICS: Dict[str, Tuple[str, ...]] = {}
"""``benchmark_name`` -> allow-list of final report metric names generated at runtime.

Populated by task 5.5 from the audit ``dynamic`` bucket (``pass@{k}`` families and f-string
composed names)."""


def _validate_catalog() -> None:
    """Materialize every catalog entry at import time so illegal declarations fail fast.

    Each :class:`MetricEntry` is resolved: its ``baseline`` reference must exist in
    :data:`SEMANTIC_BASELINES` and the merged declaration must pass the contract validation.

    Raises:
        ValueError: If an entry references a baseline absent from the baseline table.
        pydantic.ValidationError: If a resolved entry violates the metric semantics contract.
    """
    for name, entry in METRIC_NAME_SEMANTICS.items():
        if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES:
            raise ValueError(
                f"metric name '{name}' in {METRIC_NAME_TABLE_LOCATION} references unknown baseline "
                f"'{entry.baseline}'; declare it at {BASELINE_TABLE_LOCATION}"
            )
        entry.resolve(name)

    for (benchmark_name, metric_name), entry in BENCHMARK_METRIC_OVERRIDES.items():
        if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES:
            raise ValueError(
                f"override ('{benchmark_name}', '{metric_name}') in {BENCHMARK_OVERRIDE_TABLE_LOCATION} "
                f"references unknown baseline '{entry.baseline}'; declare it at {BASELINE_TABLE_LOCATION}"
            )
        entry.resolve(metric_name)


_validate_catalog()


def lookup_metric_entry(final_metric_name: str) -> MetricEntry:
    """Look up the catalog entry of a final report metric name, or ``None`` when undeclared.

    Exact-key lookup against :data:`METRIC_NAME_SEMANTICS`: no name normalization and no fuzzy
    matching, matching the resolution rules.

    Args:
        final_metric_name: Final report metric name as written into ``Metric.name``.

    Returns:
        The declared entry, or ``None`` when the name is not in the catalog.
    """
    return METRIC_NAME_SEMANTICS.get(final_metric_name)


__all__ = [
    'BENCHMARK_DYNAMIC_METRICS',
    'BENCHMARK_METRIC_OVERRIDES',
    'BENCHMARK_OVERRIDE_TABLE_LOCATION',
    'DYNAMIC_METRIC_TABLE_LOCATION',
    'METRIC_NAME_SEMANTICS',
    'METRIC_NAME_TABLE_LOCATION',
    'lookup_metric_entry',
]
