# Copyright (c) Alibaba, Inc. and its affiliates.
"""Thread-safe performance metrics collector for the evaluation pipeline.

PerfCollector is instantiated by DefaultEvaluator at startup.  It accumulates
per-sample PerformanceMetrics objects produced during model inference and
exposes summary statistics and JSON export helpers.
"""
import os
import pandas as pd
from typing import Any, Dict, List, Optional

from evalscope.api.model.perf_metrics import PerformanceMetrics, PerfSummary
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.io_utils import dict_to_json
from evalscope.utils.logger import get_logger

logger = get_logger()

_PERCENTILES = [0.25, 0.5, 0.75, 0.9, 0.99]


def _series_stats(s: pd.Series) -> dict:
    """Compute stats using pd.Series.describe() with custom percentiles.

    Returns keys: mean / std / min / 25% / 50% / 75% / 90% / 99% / max,
    all rounded to 6 decimal places.  std uses pandas default ddof=1.
    """
    desc = s.describe(percentiles=_PERCENTILES)
    keys = ['mean', 'std', 'min', '25%', '50%', '75%', '90%', '99%', 'max']
    return {k: round(float(desc[k]), 6) for k in keys}


class PerfCollector:
    """Thread-safe collector for per-sample performance metrics during evaluation.

    Usage::

        collector = PerfCollector()
        # (called from multiple threads inside the eval pool)
        collector.record(task_state.output.perf_metrics)
        # (called once from the main thread after the pool finishes)
        summary = collector.get_summary()
        collector.save_report('/path/to/reports/model_name', 'benchmark.json')
    """

    def __init__(self) -> None:
        self._samples: List[PerformanceMetrics] = []

    # ------------------------------------------------------------------ #
    # Data ingestion                                                       #
    # ------------------------------------------------------------------ #

    @thread_safe
    def record(self, perf: PerformanceMetrics) -> None:
        """Append one sample's metrics to the internal store (thread-safe).

        Args:
            perf: A :class:`PerformanceMetrics` instance from a completed
                inference call.
        """
        self._samples.append(perf)

    @thread_safe
    def _get_samples(self) -> List[PerformanceMetrics]:
        """Return a snapshot of collected samples (thread-safe)."""
        return list(self._samples)

    # ------------------------------------------------------------------ #
    # Aggregation                                                          #
    # ------------------------------------------------------------------ #

    def get_summary(self) -> Optional[PerfSummary]:
        """Compute aggregate statistics across all recorded samples.

        Returns:
            A :class:`PerfSummary` instance with avg/min/max/std, throughput,
            token usage, and percentile breakdowns for each available metric.
            ``None`` when no samples have been recorded.
        """
        samples = self._get_samples()
        if not samples:
            return None

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
        """Return a list of per-sample metric dictionaries.

        Each entry has ``sample_index`` (0-based insertion order), plus all
        fields from :class:`PerformanceMetrics`.
        """
        samples = self._get_samples()

        records = []
        for i, s in enumerate(samples):
            record: Dict[str, Any] = {
                'sample_index': i,
                'latency': s.latency,
                'input_tokens': s.input_tokens,
                'output_tokens': s.output_tokens,
            }
            if s.ttft is not None:
                record['ttft'] = s.ttft
            if s.tpot is not None:
                record['tpot'] = s.tpot
            records.append(record)
        return records

    def get_perf_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for embedding in a report JSON.

        Returns:
            Dict with ``summary`` key (nested structure by metric category), or
            empty dict when no samples have been collected.
        """
        summary = self.get_summary()
        if summary is None:
            return {}
        return {'summary': summary.to_dict()}

    def save_report(self, output_dir: str, filename: str = 'perf_metrics.json') -> Optional[str]:
        """Write summary statistics to a JSON file.

        Args:
            output_dir: Directory in which to write the report.  Created
                automatically if it does not exist.
            filename: Base name for the output file.

        Returns:
            Absolute path of the written file, or ``None`` if there are no
            samples to report.
        """
        report = self.get_perf_dict()
        if not report:
            logger.debug('PerfCollector: no samples recorded, skipping report.')
            return None

        report_path = os.path.join(output_dir, filename)
        dict_to_json(report, report_path)
        logger.info(f'Performance metrics report saved to: {report_path}')
        return report_path
