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
    TIME_TAKEN_FOR_TESTS = 'Time taken for tests (s)'
    NUMBER_OF_CONCURRENCY = 'Number of concurrency'
    REQUEST_RATE = 'Request rate (req/s)'
    TOTAL_REQUESTS = 'Total requests'
    SUCCEED_REQUESTS = 'Succeed requests'
    FAILED_REQUESTS = 'Failed requests'
    REQUEST_THROUGHPUT = 'Request throughput (req/s)'
    AVERAGE_LATENCY = 'Average latency (s)'
    AVERAGE_INPUT_TOKENS_PER_REQUEST = 'Average input tokens per request'

    # LLM-specific
    OUTPUT_TOKEN_THROUGHPUT = 'Output token throughput (tok/s)'
    TOTAL_TOKEN_THROUGHPUT = 'Total token throughput (tok/s)'
    AVERAGE_TIME_TO_FIRST_TOKEN = 'Average time to first token (s)'
    AVERAGE_TIME_PER_OUTPUT_TOKEN = 'Average time per output token (s)'
    AVERAGE_INTER_TOKEN_LATENCY = 'Average inter-token latency (s)'
    AVERAGE_OUTPUT_TOKENS_PER_REQUEST = 'Average output tokens per request'

    # Embedding / Rerank-specific
    INPUT_TOKEN_THROUGHPUT = 'Input token throughput (tok/s)'

    # Multi-turn specific
    AVERAGE_INPUT_TURNS_PER_REQUEST = 'Average input turns per request'
    AVERAGE_CACHED_PERCENT = 'Average approx KV cache hit rate (%)'

    # Speculative decoding specific
    AVERAGE_DECODED_TOKENS_PER_ITER = 'Average decoded tokens per iter (tok/iter)'
    APPROX_SPECULATIVE_ACCEPTANCE_RATE = 'Approx speculative decoding acceptance rate'

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

    TTFT = 'TTFT (s)'
    ITL = 'ITL (s)'
    TPOT = 'TPOT (s)'
    LATENCY = 'Latency (s)'
    INPUT_TOKENS = 'Input tokens'
    OUTPUT_TOKENS = 'Output tokens'
    OUTPUT_THROUGHPUT = 'Output (tok/s)'
    INPUT_THROUGHPUT = 'Input (tok/s)'
    TOTAL_THROUGHPUT = 'Total (tok/s)'
    PERCENTILES = 'Percentiles'
