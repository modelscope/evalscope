"""Centralised metric name constants for the perf benchmark pipeline.

Both ``Metrics`` (summary-level) and ``PercentileMetrics`` (percentile-level)
live here so that every layer — data collection, DB storage, rich display,
and HTML report generation — shares a single source of truth for field names.

All other modules should import from here instead of defining their own
string literals:

    from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Summary-level metric names  (keys in benchmark_summary.json)
# ---------------------------------------------------------------------------


class Metrics:
    """Standardized metric name constants used throughout the benchmark pipeline.

    Also provides API-type classification via :meth:`is_embedding_or_rerank`.
    """

    # General
    TIME_TAKEN_FOR_TESTS = 'Test Duration (s)'
    NUMBER_OF_CONCURRENCY = 'Concurrency'
    REQUEST_RATE = 'Request Rate (req/s)'
    TOTAL_REQUESTS = 'Total Requests'
    SUCCEED_REQUESTS = 'Success Requests'
    FAILED_REQUESTS = 'Failed Requests'
    REQUEST_THROUGHPUT = 'Req Throughput (req/s)'
    AVERAGE_LATENCY = 'Avg Latency (s)'
    AVERAGE_INPUT_TOKENS_PER_REQUEST = 'Avg Input Tokens'

    # LLM-specific
    OUTPUT_TOKEN_THROUGHPUT = 'Output Throughput (tok/s)'
    TOTAL_TOKEN_THROUGHPUT = 'Total Throughput (tok/s)'
    AVERAGE_TIME_TO_FIRST_TOKEN = 'TTFT (ms)'
    AVERAGE_TIME_PER_OUTPUT_TOKEN = 'TPOT (ms)'
    AVERAGE_INTER_TOKEN_LATENCY = 'ITL (ms)'
    AVERAGE_OUTPUT_TOKENS_PER_REQUEST = 'Avg Output Tokens'

    # Embedding / Rerank-specific
    INPUT_TOKEN_THROUGHPUT = 'Input Throughput (tok/s)'

    # Multi-turn specific
    AVERAGE_INPUT_TURNS_PER_REQUEST = 'Avg Turns/Request'
    AVERAGE_CACHED_PERCENT = 'KV Cache Hit Rate (%)'

    # Speculative decoding specific
    AVERAGE_DECODED_TOKENS_PER_ITER = 'Decoded Tok/Iter'
    APPROX_SPECULATIVE_ACCEPTANCE_RATE = 'Spec. Accept Rate'

    @staticmethod
    def is_embedding_or_rerank(api_name: str) -> bool:
        """Return True if *api_name* refers to an embedding or rerank API."""
        if api_name is None:
            return False
        api_lower = api_name.lower()
        return 'embedding' in api_lower or 'rerank' in api_lower or 'embed' in api_lower


# ---------------------------------------------------------------------------
# Percentile-level metric names  (keys in benchmark_percentile.json)
# ---------------------------------------------------------------------------


class PercentileMetrics:
    """Metric name constants for the percentile results table.

    These are the column keys used in ``benchmark_percentile.json`` and in the
    ``get_percentile_results`` function output.
    """

    TTFT = 'TTFT (ms)'
    ITL = 'ITL (ms)'
    TPOT = 'TPOT (ms)'
    LATENCY = 'Latency (s)'
    INPUT_TOKENS = 'Input tokens'
    OUTPUT_TOKENS = 'Output tokens'
    OUTPUT_THROUGHPUT = 'Output (tok/s)'
    INPUT_THROUGHPUT = 'Input (tok/s)'
    TOTAL_THROUGHPUT = 'Total (tok/s)'
    DECODE_THROUGHPUT = 'Decode (tok/s)'
    PERCENTILES = 'Percentiles'
