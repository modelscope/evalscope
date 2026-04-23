# Copyright (c) Alibaba, Inc. and its affiliates.
"""Thread-safe performance metrics collector for the evaluation pipeline.

PerfCollector is instantiated by DefaultEvaluator at startup.  It accumulates
per-sample PerformanceMetrics objects produced during model inference and
exposes summary statistics and JSON export helpers.
"""
import os
import statistics
from typing import Any, Dict, List, Optional

from evalscope.api.model.perf_metrics import PerformanceMetrics
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.io_utils import dict_to_json
from evalscope.utils.logger import get_logger

logger = get_logger()


def _safe_avg(values: List[float]) -> Optional[float]:
    """Return the mean of *values*, or ``None`` if the list is empty."""
    return sum(values) / len(values) if values else None


def _percentiles(values: List[float], ps: List[int]) -> Dict[str, float]:
    """Compute percentiles for a sorted list of values."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in ps:
        idx = min(int(n * p / 100), n - 1)
        result[f'p{p}'] = round(sorted_vals[idx], 6)
    return result


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

    _PERCENTILE_PS = [50, 75, 90, 95, 99]

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
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all recorded samples.

        Returns:
            A dictionary with average, min/max/std, throughput and percentile
            breakdowns for each available metric.  Empty dict when no samples
            have been recorded.
        """
        samples = self._get_samples()

        if not samples:
            return {}

        n = len(samples)
        latencies = [s.latency for s in samples]
        ttfts = [s.ttft for s in samples if s.ttft is not None]
        tpots = [s.tpot for s in samples if s.tpot is not None]
        input_tokens = [s.input_tokens for s in samples]
        output_tokens = [s.output_tokens for s in samples]

        avg_latency = _safe_avg(latencies) or 0.0
        avg_output_tokens = _safe_avg(output_tokens) or 0.0

        summary: Dict[str, Any] = {
            'n_samples': n,
            # Latency statistics
            'avg_latency': round(avg_latency, 6),
            'min_latency': round(min(latencies), 6),
            'max_latency': round(max(latencies), 6),
            'std_latency': round(statistics.pstdev(latencies), 6),
            # TTFT / TPOT averages
            'avg_ttft': round(_safe_avg(ttfts), 6) if ttfts else None,
            'avg_tpot': round(_safe_avg(tpots), 6) if tpots else None,
            # Token statistics
            'avg_input_tokens': round(_safe_avg(input_tokens) or 0.0, 2),
            'avg_output_tokens': round(avg_output_tokens, 2),
            'avg_throughput': round(avg_output_tokens / avg_latency, 2) if avg_latency > 0 else 0.0,
            'total_input_tokens': sum(input_tokens),
            'total_output_tokens': sum(output_tokens),
            # Percentile breakdowns
            'latency_percentiles': _percentiles(latencies, self._PERCENTILE_PS),
        }

        if ttfts:
            summary['min_ttft'] = round(min(ttfts), 6)
            summary['max_ttft'] = round(max(ttfts), 6)
            summary['ttft_percentiles'] = _percentiles(ttfts, self._PERCENTILE_PS)
        if tpots:
            summary['min_tpot'] = round(min(tpots), 6)
            summary['max_tpot'] = round(max(tpots), 6)
            summary['tpot_percentiles'] = _percentiles(tpots, self._PERCENTILE_PS)

        return summary

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
            Dict with ``summary`` key, or empty dict when no samples have been
            collected.  Per-sample records are intentionally omitted here as
            each prediction entry in the output file already carries the same
            information.
        """
        summary = self.get_summary()
        if not summary:
            return {}
        return {'summary': summary}

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
