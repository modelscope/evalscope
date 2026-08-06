"""Perf field semantics.

``PERF_FIELD_SEMANTICS`` declares the direction, unit and display rules of the public perf
contract. Keys are taken from the ``Metrics`` / ``PercentileMetrics`` constants rather than
written as literals, so renaming a constant cannot silently orphan an entry.

Directions follow what the field measures, not how it reads:

* latency style fields (test duration, TTFT, TPOT, ITL, end-to-end latency) are
  ``lower_is_better``
* throughput style fields (tokens or requests per second) are ``higher_is_better``
* request counts, token counts, cache and speculative-decoding details and the concurrency /
  request-rate knobs are ``diagnostic``: they describe the run, they are not better when larger

The entries are only attached to service API responses. No perf JSON written to disk changes
shape, and no numeric value is touched.
"""

from typing import Dict

from evalscope.api.metric.semantics import MetricEntry, MetricRole
from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics

PERF_FIELD_SEMANTICS: Dict[str, MetricEntry] = {
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

PERF_API_PATH_SEMANTICS: Dict[str, MetricEntry] = {
    # In-report `perf_metrics` and the perf run list use stable API paths rather than the display
    # names of the archive tables, so they need their own key set. Both wire shapes stay as they
    # are; only the semantics map is added next to them.
    'latency': MetricEntry(baseline='perf.latency.seconds', metric_name='Latency'),
    'best_latency': MetricEntry(baseline='perf.latency.seconds', metric_name='Best Latency'),
    'ttft': MetricEntry(baseline='perf.latency.seconds', metric_name='TTFT'),
    'tpot': MetricEntry(baseline='perf.latency.seconds', metric_name='TPOT'),
    'throughput.avg_output_tps': MetricEntry(
        baseline='perf.throughput.tokens_per_second',
        metric_name='Output Throughput',
    ),
    'throughput.avg_req_ps': MetricEntry(
        baseline='perf.throughput.requests_per_second',
        metric_name='Request Throughput',
    ),
    'best_rps': MetricEntry(baseline='perf.throughput.requests_per_second', metric_name='Best RPS'),
    'success_rate': MetricEntry(
        baseline='quality.accuracy.ratio',
        metric_name='Success Rate',
        role=MetricRole.AUXILIARY,
    ),
    'usage.input_tokens': MetricEntry(baseline='diagnostic.count.items', metric_name='Input Tokens'),
    'usage.output_tokens': MetricEntry(baseline='diagnostic.count.items', metric_name='Output Tokens'),
    'usage.total_tokens': MetricEntry(baseline='diagnostic.count.items', metric_name='Total Tokens'),
    'n_samples': MetricEntry(baseline='diagnostic.count.items', metric_name='Samples'),
}
"""Stable API path -> catalog entry, for perf data exposed under paths instead of display names."""

__all__ = ['PERF_API_PATH_SEMANTICS', 'PERF_FIELD_SEMANTICS']
