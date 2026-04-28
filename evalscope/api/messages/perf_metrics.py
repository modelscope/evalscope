# Copyright (c) Alibaba, Inc. and its affiliates.
"""Performance metrics data structures for per-request inference profiling.

PerformanceMetrics is attached to ChatMessage (assistant role) when performance
collection is enabled (TaskConfig.collect_perf=True), capturing latency, TTFT,
and token usage for a single generation call.

PerfSummary is the aggregated counterpart produced by PerfCollector after all
requests have been collected.  It carries nested statistics by metric category
(latency, throughput, usage, ttft, tpot) and exposes convenience properties
for reporting.
"""
from dataclasses import dataclass
from pydantic import BaseModel, computed_field
from typing import Optional


class PerformanceMetrics(BaseModel):
    """Per-request performance metrics collected during model inference.

    Populated by model API implementations when ``collect_perf`` is enabled.
    All timing fields are in seconds.  Token counts mirror :class:`ModelUsage`.

    Attributes:
        latency: Total end-to-end request latency in seconds
            (mirrors ``ModelOutput.time``).
        ttft: Time To First Token in seconds.  Only available when streaming
            is enabled.  Measured from request start until the first
            non-empty content chunk arrives.
        tpot: Computed — Time Per Output Token in seconds (average).  Derived
            as ``(latency - ttft) / (output_tokens - 1)`` when both values
            are available and ``output_tokens > 1``.
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of generated (completion) tokens.
    """

    latency: float = 0.0
    """Total end-to-end request latency in seconds."""

    ttft: Optional[float] = None
    """Time To First Token in seconds (streaming only)."""

    input_tokens: int = 0
    """Number of input (prompt) tokens."""

    output_tokens: int = 0
    """Number of generated (completion) tokens."""

    @computed_field
    @property
    def tpot(self) -> Optional[float]:
        """Average Time Per Output Token in seconds (derived, not stored)."""
        return calc_tpot(self.latency, self.ttft, self.output_tokens)


def calc_tpot(latency: float, ttft: Optional[float], output_tokens: int) -> Optional[float]:
    """Derive average Time Per Output Token.

    Args:
        latency: Total request latency in seconds.
        ttft: Time to first token in seconds, or ``None`` if not available.
        output_tokens: Number of generated tokens.

    Returns:
        TPOT in seconds, or ``None`` when the formula cannot be applied
        (missing TTFT, fewer than two output tokens, or negative decode time).
        Returns ``0.0`` when all tokens arrived in the first chunk
        (``latency == ttft`` with ``output_tokens > 1``).
    """
    if ttft is None or output_tokens <= 1:
        return None
    decode_time = latency - ttft
    if decode_time < 0:
        return None
    return decode_time / (output_tokens - 1)


@dataclass
class PerfSummary:
    """Structured container for aggregated performance metrics.

    Produced by :class:`~evalscope.evaluator.perf_collector.PerfCollector`
    after all requests have been collected.  Provides typed properties for
    convenient access in reporting code, plus serialization helpers for JSON
    persistence.

    Each stats dict (latency / ttft / tpot / usage sub-keys) shares the same
    shape returned by ``pd.Series.describe()``:
    ``mean / std / min / 25% / 50% / 75% / 90% / 99% / max``.

    Attributes:
        n_samples:  Total number of recorded inference requests.
        latency:    Latency stats dict (seconds).
        throughput: Throughput dict with avg_output_tps (tok/s) and avg_req_ps (req/s).
        usage:      Token usage dict; each sub-key (input_tokens / output_tokens /
                    total_tokens) has the same shape as ``latency``.
        ttft:       TTFT stats dict (same shape as ``latency``), or ``None`` when
                    TTFT was not measured.
        tpot:       TPOT stats dict (same shape as ``latency``), or ``None`` when
                    TPOT was not measured.
    """

    n_samples: int
    latency: dict
    throughput: dict
    usage: dict
    ttft: Optional[dict] = None
    tpot: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Convenience properties for reporting                                 #
    # ------------------------------------------------------------------ #

    @property
    def avg_latency(self) -> float:
        """Average end-to-end latency in seconds."""
        return self.latency.get('mean', 0.0)

    @property
    def avg_ttft(self) -> Optional[float]:
        """Average time-to-first-token in seconds, or ``None``."""
        return (self.ttft or {}).get('mean')

    @property
    def avg_tpot(self) -> Optional[float]:
        """Average time-per-output-token in seconds, or ``None``."""
        return (self.tpot or {}).get('mean')

    @property
    def avg_output_tps(self):
        """Average output throughput in tokens/second."""
        return self.throughput.get('avg_output_tps', '-')

    @property
    def avg_req_ps(self):
        """Average request throughput in requests/second."""
        return self.throughput.get('avg_req_ps', '-')

    @property
    def avg_input_tokens(self):
        """Average number of input tokens per request."""
        return self.usage.get('input_tokens', {}).get('mean', '-')

    @property
    def avg_output_tokens(self):
        """Average number of output tokens per request."""
        return self.usage.get('output_tokens', {}).get('mean', '-')

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialize to a nested dict suitable for JSON report storage."""
        d: dict = {
            'n_samples': self.n_samples,
            'latency': self.latency,
            'throughput': self.throughput,
            'usage': self.usage,
        }
        if self.ttft is not None:
            d['ttft'] = self.ttft
        if self.tpot is not None:
            d['tpot'] = self.tpot
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'PerfSummary':
        """Reconstruct a PerfSummary from a JSON-loaded summary dict."""
        return cls(
            n_samples=d.get('n_samples', 0),
            latency=d.get('latency', {}),
            throughput=d.get('throughput', {}),
            usage=d.get('usage', {}),
            ttft=d.get('ttft'),
            tpot=d.get('tpot'),
        )
