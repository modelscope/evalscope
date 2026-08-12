"""Perf field semantics.

``PERF_SEMANTICS`` declares the direction, unit and display rules of the public perf
contract. Keys are taken from the ``Metrics`` / ``PercentileMetrics`` constants rather than
written as literals, so renaming a constant cannot silently orphan an entry.

Directions follow what the field measures, not how it reads:

* latency style fields (test duration, TTFT, TPOT, ITL, end-to-end latency) are
  ``lower_is_better``
* throughput style fields (tokens or requests per second) are ``higher_is_better``
* request counts, token counts, cache and speculative-decoding details and the concurrency /
  request-rate knobs are ``diagnostic``: they describe the run, they are not better when larger

Report v2 persists the resolved entries next to embedded perf values; perf archive APIs resolve
the same registry for their own wire shapes. No numeric value is touched.

This module is data only, so ``resolver`` can read it at import time. The functions that bind
these entries to a service payload live in ``resolver`` and are re-exported from the package.
"""

from typing import Dict

from evalscope.api.metric.semantics import MetricRole
from evalscope.metrics.semantics.entry import MetricEntry
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics

PERF_SEMANTICS: Dict[str, MetricEntry] = {
    # --- run shape: knobs and counts, no direction ----------------------------------------
    Metrics.TIME_TAKEN_FOR_TESTS: MetricEntry(
        baseline='diagnostic.unspecified',
        metric_name=Metrics.TIME_TAKEN_FOR_TESTS,
        raw_unit='s',
        display_unit='s',
        display_precision=2,
    ),
    Metrics.NUMBER_OF_CONCURRENCY: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.NUMBER_OF_CONCURRENCY,
    ),
    Metrics.REQUEST_RATE: MetricEntry(
        baseline='diagnostic.unspecified',
        metric_name=Metrics.REQUEST_RATE,
        raw_unit='req/s',
        display_unit='req/s',
        display_precision=2,
    ),
    Metrics.TOTAL_REQUESTS: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.TOTAL_REQUESTS,
    ),
    Metrics.SUCCEED_REQUESTS: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.SUCCEED_REQUESTS,
    ),
    Metrics.FAILED_REQUESTS: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.FAILED_REQUESTS,
    ),
    Metrics.STREAM_REQUESTS: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.STREAM_REQUESTS,
    ),
    Metrics.NON_STREAM_REQUESTS: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.NON_STREAM_REQUESTS,
    ),
    # --- throughput: higher is better -----------------------------------------------------
    Metrics.REQUEST_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.requests_per_second',
        metric_name=Metrics.REQUEST_THROUGHPUT,
    ),
    Metrics.OUTPUT_TOKEN_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name=Metrics.OUTPUT_TOKEN_THROUGHPUT,
    ),
    Metrics.TOTAL_TOKEN_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name=Metrics.TOTAL_TOKEN_THROUGHPUT,
    ),
    Metrics.INPUT_TOKEN_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name=Metrics.INPUT_TOKEN_THROUGHPUT,
    ),
    # --- latency: lower is better ---------------------------------------------------------
    Metrics.AVERAGE_LATENCY: MetricEntry(
        baseline='perf.latency.seconds',
        metric_name=Metrics.AVERAGE_LATENCY,
    ),
    Metrics.AVERAGE_TIME_TO_FIRST_TOKEN: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=Metrics.AVERAGE_TIME_TO_FIRST_TOKEN,
    ),
    Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN,
    ),
    Metrics.AVERAGE_INTER_TOKEN_LATENCY: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=Metrics.AVERAGE_INTER_TOKEN_LATENCY,
    ),
    Metrics.AVERAGE_FIRST_TURN_TTFT: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=Metrics.AVERAGE_FIRST_TURN_TTFT,
    ),
    Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT,
    ),
    # --- token and turn volume: describes the workload, not its quality -------------------
    Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST,
        display_precision=1,
    ),
    Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST,
        display_precision=1,
    ),
    Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST,
        display_precision=1,
    ),
    # --- cache and speculative decoding details -------------------------------------------
    #: Already expressed as a percentage by the pipeline, hence multiplier 1.
    Metrics.AVERAGE_CACHED_PERCENT: MetricEntry(
        baseline='diagnostic.parse_status.ratio',
        metric_name=Metrics.AVERAGE_CACHED_PERCENT,
        value_range={
            'min': 0.0,
            'max': 100.0
        },
        display_multiplier=1.0,
    ),
    Metrics.AVERAGE_DECODED_TOKENS_PER_ITER: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=Metrics.AVERAGE_DECODED_TOKENS_PER_ITER,
        display_precision=2,
    ),
    Metrics.APPROX_SPECULATIVE_ACCEPTANCE_RATE: MetricEntry(
        baseline='diagnostic.unspecified',
        metric_name=Metrics.APPROX_SPECULATIVE_ACCEPTANCE_RATE,
        display_precision=3,
    ),
    # --- percentile table: same directions as their summary counterparts ------------------
    PercentileMetrics.TTFT: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=PercentileMetrics.TTFT,
    ),
    PercentileMetrics.ITL: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=PercentileMetrics.ITL,
    ),
    PercentileMetrics.TPOT: MetricEntry(
        baseline='perf.latency.milliseconds',
        metric_name=PercentileMetrics.TPOT,
    ),
    PercentileMetrics.LATENCY: MetricEntry(
        baseline='perf.latency.seconds',
        metric_name=PercentileMetrics.LATENCY,
    ),
    PercentileMetrics.OUTPUT_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name=PercentileMetrics.OUTPUT_THROUGHPUT,
    ),
    PercentileMetrics.INPUT_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name=PercentileMetrics.INPUT_THROUGHPUT,
    ),
    PercentileMetrics.TOTAL_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name=PercentileMetrics.TOTAL_THROUGHPUT,
    ),
    PercentileMetrics.DECODE_THROUGHPUT: MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name=PercentileMetrics.DECODE_THROUGHPUT,
    ),
    PercentileMetrics.INPUT_TOKENS: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=PercentileMetrics.INPUT_TOKENS,
    ),
    PercentileMetrics.OUTPUT_TOKENS: MetricEntry(
        baseline='diagnostic.count.items',
        metric_name=PercentileMetrics.OUTPUT_TOKENS,
    ),
    #: The percentile label column itself, carried along so the table has no unlabelled key.
    PercentileMetrics.PERCENTILES: MetricEntry(
        baseline='diagnostic.unspecified',
        metric_name=PercentileMetrics.PERCENTILES,
        display_precision=0,
    ),
}
"""Perf field key -> catalog entry. Keys come from the perf name constants."""

PERF_SEMANTICS.update({
    # In-report `perf_metrics` and the perf run list expose their numbers under stable API paths
    # rather than under the display names of the archive tables, so they need their own key set.
    # Both wire shapes stay as they are; only the semantics map is added next to them.
    'latency': MetricEntry(baseline='perf.latency.seconds', metric_name='Latency'),
    'best_latency': MetricEntry(baseline='perf.latency.seconds', metric_name='Best Latency'),
    'ttft': MetricEntry(
        baseline='perf.latency.seconds',
        metric_name='TTFT',
        display_multiplier=1000.0,
        display_unit='ms',
        display_precision=1,
    ),
    'tpot': MetricEntry(
        baseline='perf.latency.seconds',
        metric_name='TPOT',
        display_multiplier=1000.0,
        display_unit='ms',
        display_precision=1,
    ),
    'throughput.avg_output_tps': MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name='Output Throughput',
    ),
    'throughput.avg_req_ps': MetricEntry(
        baseline='perf.throughput.requests_per_second',
        metric_name='Request Throughput',
    ),
    #: The display name already says RPS, so repeating `req/s` after the value only adds noise
    #: wherever the two are shown together. `raw_unit` keeps the unit for tooltips and exports.
    #: An empty string rather than `None`: a `None` override is skipped by `MetricEntry.resolve`,
    #: which would leave the baseline's unit in place.
    'best_rps': MetricEntry(
        baseline='perf.throughput.requests_per_second',
        metric_name='Best RPS',
        display_unit='',
    ),
    #: The run list reports this as a 0-100 percentage, so it renders with multiplier 1.
    'success_rate': MetricEntry(
        baseline='quality.score.points_100',
        metric_name='Success Rate',
        role=MetricRole.AUXILIARY,
    ),
    'usage.input_tokens': MetricEntry(baseline='diagnostic.count.items', metric_name='Input Tokens'),
    'usage.output_tokens': MetricEntry(baseline='diagnostic.count.items', metric_name='Output Tokens'),
    'usage.total_tokens': MetricEntry(baseline='diagnostic.count.items', metric_name='Total Tokens'),
    'n_samples': MetricEntry(baseline='diagnostic.count.items', metric_name='Samples'),
})
"""Stable API path -> catalog entry, for perf data exposed under paths instead of display names."""

PERF_SEMANTICS.update({
    # The archive's cross-run summary table labels its columns for humans (`build_summary_table`),
    # so those labels are a third key space next to the perf constants and the API paths. Values
    # follow the formatting of that table: latencies in seconds, TTFT / TPOT in milliseconds and
    # the success rate already scaled to 0-100.
    'RPS': MetricEntry(baseline='perf.throughput.requests_per_second', metric_name='RPS'),
    'Avg Lat.(s)': MetricEntry(baseline='perf.latency.seconds', metric_name='Avg Latency'),
    'P99 Lat.(s)': MetricEntry(baseline='perf.latency.seconds', metric_name='P99 Latency'),
    'Avg TTFT(ms)': MetricEntry(baseline='perf.latency.milliseconds', metric_name='Avg TTFT'),
    'P99 TTFT(ms)': MetricEntry(baseline='perf.latency.milliseconds', metric_name='P99 TTFT'),
    'Avg TPOT(ms)': MetricEntry(baseline='perf.latency.milliseconds', metric_name='Avg TPOT'),
    'P99 TPOT(ms)': MetricEntry(baseline='perf.latency.milliseconds', metric_name='P99 TPOT'),
    'Gen. tok/s': MetricEntry(baseline='perf.throughput.tokens_per_second', metric_name='Output Throughput'),
    'Avg Inp.TPS': MetricEntry(baseline='perf.throughput.tokens_per_second', metric_name='Input Throughput'),
    'Avg Inp.Tok': MetricEntry(
        baseline='diagnostic.count.items',
        metric_name='Avg Input Tokens',
        display_precision=1,
    ),
    #: Rendered as `87.5%` by the table, i.e. already a percentage.
    'Success Rate': MetricEntry(
        baseline='quality.score.points_100',
        metric_name='Success Rate',
        role=MetricRole.AUXILIARY,
    ),
    #: Run configuration, not a measurement: it describes the workload rather than grading it.
    'Conc.': MetricEntry(baseline='diagnostic.count.items', metric_name='Concurrency'),
    'Rate': MetricEntry(baseline='diagnostic.unspecified', metric_name='Request Rate', display_precision=2),
})
"""Archive summary table column label -> catalog entry."""

__all__ = ['PERF_SEMANTICS']
