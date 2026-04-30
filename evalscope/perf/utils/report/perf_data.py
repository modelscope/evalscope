"""Data loading utilities for perf benchmark HTML reports.

Provides:
  - RunLoader  class: discovers and loads runs from an output directory

Data models (BenchmarkSummary, PercentileResult, RequestRecord, RunData)
are defined in :mod:`evalscope.perf.utils.perf_models` and re-exported here
for backward compatibility.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import List, Optional

from evalscope.perf.utils.perf_models import BenchmarkSummary, PercentileResult, RequestRecord, RunData
from evalscope.utils.logger import get_logger

# Re-export for backward compatibility
__all__ = [
    'BenchmarkSummary',
    'PercentileResult',
    'RequestRecord',
    'RunData',
    'RunLoader',
]

logger = get_logger()

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# Patterns for recognized run directory names
_PARALLEL_RE = re.compile(r'^parallel_(\d+)_number_(\d+)$')
_RATE_RE = re.compile(r'^rate_([\d.]+)_number_(\d+)$')


class RunLoader:
    """Discovers and loads all benchmark runs from an output directory.

    Recognized sub-directory patterns:

    * ``parallel_<N>_number_<M>`` – closed-loop / fixed-concurrency mode
    * ``rate_<R>_number_<M>``     – open-loop / fixed-rate mode
    """

    @staticmethod
    def load_all(output_dir: str) -> List[RunData]:
        """Walk *output_dir* for run subdirectories and return sorted runs."""
        if not os.path.isdir(output_dir):
            return []

        runs: List[RunData] = []
        for entry in sorted(os.listdir(output_dir)):
            run_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            if not (_PARALLEL_RE.match(entry) or _RATE_RE.match(entry)):
                continue
            run = RunLoader._load_single(run_dir, entry)
            if run is not None:
                runs.append(run)

        return sorted(runs, key=lambda r: (r.parallel, r.number))

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_single(run_dir: str, dir_name: str) -> Optional[RunData]:
        summary_dict = RunLoader._load_json(os.path.join(run_dir, 'benchmark_summary.json')) or {}
        percentile_rows = RunLoader._load_json(os.path.join(run_dir, 'benchmark_percentile.json')) or []
        args = RunLoader._load_json(os.path.join(run_dir, 'benchmark_args.json')) or {}
        requests = RunLoader._load_db(run_dir)

        summary = BenchmarkSummary.from_dict(summary_dict)
        percentiles = PercentileResult.from_list(percentile_rows) if isinstance(percentile_rows, list) \
            else PercentileResult.from_transposed(percentile_rows)

        m_parallel = _PARALLEL_RE.match(dir_name)
        m_rate = _RATE_RE.match(dir_name)
        if m_parallel:
            parallel, number = int(m_parallel.group(1)), int(m_parallel.group(2))
        elif m_rate:
            # open-loop mode: encode rate*1000 as a synthetic "parallel" key so
            # that the existing (parallel, number) sort order stays meaningful.
            parallel = round(float(m_rate.group(1)) * 1000)
            number = int(m_rate.group(2))
        else:
            parallel = summary.concurrency
            number = summary.total_requests

        return RunData(
            dir_name=dir_name,
            parallel=parallel,
            number=number,
            summary=summary,
            percentiles=percentiles,
            args=args,
            requests=requests,
        )

    @staticmethod
    def _load_json(path: str):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return None

    @staticmethod
    def _load_db(run_dir: str) -> List[RequestRecord]:
        db_path = os.path.join(run_dir, 'benchmark_data.db')
        if not os.path.exists(db_path):
            return []

        records: List[RequestRecord] = []
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT start_time, completed_time, latency, first_chunk_latency, '
                    'prompt_tokens, completion_tokens, inter_token_latencies, '
                    'time_per_output_token, success FROM result'
                )
                for row in cursor.fetchall():
                    d = dict(row)
                    itl_raw = d.get('inter_token_latencies')
                    try:
                        itl: List[float] = json.loads(itl_raw) if isinstance(itl_raw, str) else []
                    except (json.JSONDecodeError, TypeError):
                        itl = []

                    # DB stores all latency fields in seconds; convert to ms on load.
                    fcl = d.get('first_chunk_latency')
                    tpot = d.get('time_per_output_token')
                    records.append(
                        RequestRecord(
                            start_time=float(d.get('start_time') or 0),
                            completed_time=float(d.get('completed_time') or 0),
                            latency=float(d.get('latency') or 0),
                            first_chunk_latency=(fcl * 1000) if fcl is not None else None,
                            prompt_tokens=int(d.get('prompt_tokens') or 0),
                            completion_tokens=int(d.get('completion_tokens') or 0),
                            inter_token_latencies=[v * 1000 for v in itl],
                            time_per_output_token=(tpot * 1000) if tpot is not None else None,
                            success=bool(d.get('success', 0)),
                        )
                    )
        except Exception as exc:
            logger.warning(f'Failed to read DB {db_path}: {exc}')

        return records
