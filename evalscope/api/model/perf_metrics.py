# Copyright (c) Alibaba, Inc. and its affiliates.
"""Performance metrics data structures for per-request inference profiling.

PerformanceMetrics is attached to ModelOutput when performance collection is
enabled (TaskConfig.collect_perf=True), capturing latency, TTFT, and token
usage for a single generation call.
"""
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
