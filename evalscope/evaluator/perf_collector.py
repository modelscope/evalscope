# Copyright (c) Alibaba, Inc. and its affiliates.
"""Thread-safe performance metrics collector for the evaluation pipeline.

PerfCollector is instantiated by DefaultEvaluator at startup.  It accumulates
per-**request** PerformanceMetrics objects produced during model inference and
exposes summary statistics and JSON export helpers.

Each ``record`` call represents a single generation request. For single-turn
benchmarks this is 1-to-1 with a sample; for multi-turn benchmarks a sample
produces one record per assistant turn, tagged with ``(sample_index,
turn_index)`` so downstream consumers can inspect / filter per-turn or
per-sample breakdowns while aggregate statistics are always computed at the
request granularity.
"""
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evalscope.api.messages.perf_metrics import PerformanceMetrics, PerfSummary
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.logger import get_logger

logger = get_logger()

_PERCENTILES = [0.25, 0.5, 0.75, 0.9, 0.99]


@dataclass
class _PerfRecord:
    """Internal container pairing a PerformanceMetrics with its identity."""
    perf: PerformanceMetrics
    sample_index: Optional[Any] = None
    turn_index: Optional[int] = None


def _series_stats(s: pd.Series) -> dict:
    """Compute stats using pd.Series.describe() with custom percentiles.

    Returns keys: mean / std / min / 25% / 50% / 75% / 90% / 99% / max,
    all rounded to 6 decimal places.  std uses pandas default ddof=1.
    """
    desc = s.describe(percentiles=_PERCENTILES)
    keys = ['mean', 'std', 'min', '25%', '50%', '75%', '90%', '99%', 'max']
    return {k: round(float(desc[k]), 6) for k in keys}


class PerfCollector:
    """Thread-safe collector for per-request performance metrics during evaluation.

    Usage::

        collector = PerfCollector()
        # single-turn: one record per sample
        collector.record(perf, sample_index=sample_id, turn_index=0)
        # multi-turn: one record per assistant message
        for t, msg in enumerate(assistant_messages):
            collector.record(msg.perf_metrics, sample_index=sample_id, turn_index=t)
        # aggregate (main thread, after the pool finishes)
        summary = collector.get_summary()
    """

    def __init__(self) -> None:
        self._records: List[_PerfRecord] = []

    # ------------------------------------------------------------------ #
    # Data ingestion                                                       #
    # ------------------------------------------------------------------ #

    @thread_safe
    def record(
        self,
        perf: PerformanceMetrics,
        sample_index: Optional[Any] = None,
        turn_index: Optional[int] = None,
    ) -> None:
        """Append one request's metrics to the internal store (thread-safe).

        Args:
            perf: A :class:`PerformanceMetrics` instance from a completed
                inference call.
            sample_index: Optional sample identifier to correlate records
                back to a specific sample (useful for multi-turn dumps).
            turn_index: Optional 0-based turn index inside a sample. ``None``
                or ``0`` for single-turn benchmarks.
        """
        self._records.append(_PerfRecord(perf=perf, sample_index=sample_index, turn_index=turn_index))

    @thread_safe
    def _get_records(self) -> List[_PerfRecord]:
        """Return a snapshot of collected records (thread-safe)."""
        return list(self._records)

    # ------------------------------------------------------------------ #
    # Aggregation                                                          #
    # ------------------------------------------------------------------ #

    def get_summary(self) -> Optional[PerfSummary]:
        """Compute aggregate statistics across all recorded requests.

        Returns:
            A :class:`PerfSummary` instance with avg/min/max/std, throughput,
            token usage, and percentile breakdowns for each available metric.
            ``None`` when no requests have been recorded.
        """
        records = self._get_records()
        if not records:
            return None

        samples = [r.perf for r in records]
        n = len(samples)

        latencies = pd.Series([s.latency for s in samples], dtype=float)
        ttfts_raw = [s.ttft for s in samples if s.ttft is not None]
        tpots_raw = [s.tpot for s in samples if s.tpot is not None]
        input_tokens = pd.Series([s.input_tokens for s in samples], dtype=float)
        output_tokens = pd.Series([s.output_tokens for s in samples], dtype=float)
        total_tokens = input_tokens + output_tokens

        avg_latency = float(latencies.mean())
        avg_output = float(output_tokens.mean())

        latency_stats = _series_stats(latencies)
        throughput = {
            'avg_output_tps': round(avg_output / avg_latency, 2) if avg_latency > 0 else 0.0,
            'avg_req_ps': round(1.0 / avg_latency, 4) if avg_latency > 0 else 0.0,
        }
        usage = {
            'input_tokens': _series_stats(input_tokens),
            'output_tokens': _series_stats(output_tokens),
            'total_tokens': _series_stats(total_tokens),
            'total_input_tokens': int(input_tokens.sum()),
            'total_output_tokens': int(output_tokens.sum()),
            'total_tokens_count': int(total_tokens.sum()),
        }
        ttft_stats = _series_stats(pd.Series(ttfts_raw, dtype=float)) if ttfts_raw else None
        tpot_stats = _series_stats(pd.Series(tpots_raw, dtype=float)) if tpots_raw else None

        return PerfSummary(
            n_samples=n,
            latency=latency_stats,
            throughput=throughput,
            usage=usage,
            ttft=ttft_stats,
            tpot=tpot_stats,
        )

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def get_per_sample_records(self) -> List[Dict[str, Any]]:
        """Return a list of per-request metric dictionaries.

        Each entry has ``sample_index`` (the sample id passed to ``record``,
        falling back to insertion order when absent), ``turn_index`` (0-based
        turn position inside the sample, ``None`` for single-turn), plus all
        fields from :class:`PerformanceMetrics`.
        """
        records = self._get_records()

        out: List[Dict[str, Any]] = []
        for i, r in enumerate(records):
            s = r.perf
            record: Dict[str, Any] = {
                'sample_index': r.sample_index if r.sample_index is not None else i,
                'turn_index': r.turn_index,
                'latency': s.latency,
                'input_tokens': s.input_tokens,
                'output_tokens': s.output_tokens,
            }
            if s.ttft is not None:
                record['ttft'] = s.ttft
            if s.tpot is not None:
                record['tpot'] = s.tpot
            out.append(record)
        return out

    def get_perf_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for embedding in a report JSON.

        Returns:
            Dict with ``summary`` key (nested structure by metric category), or
            empty dict when no requests have been collected.
        """
        summary = self.get_summary()
        if summary is None:
            return {}
        return {'summary': summary.to_dict()}
