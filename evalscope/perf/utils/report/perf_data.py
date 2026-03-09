"""Data models and loading utilities for perf benchmark HTML reports.

Provides:
  - RequestRecord  dataclass: a single per-request DB row
  - RunData        dataclass: all data for one parallel_X_number_Y run
  - RunLoader      class:     discovers and loads runs from an output directory
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RequestRecord:
    """A single benchmark request record loaded from the SQLite DB."""

    start_time: float
    completed_time: float
    latency: float
    first_chunk_latency: Optional[float]
    prompt_tokens: int
    completion_tokens: int
    inter_token_latencies: List[float]
    time_per_output_token: Optional[float]
    success: bool


@dataclasses.dataclass
class RunData:
    """All data for a single benchmark run (one parallel_X_number_Y directory)."""

    dir_name: str
    parallel: int
    number: int
    summary: Dict[str, Any]
    percentiles: List[Dict[str, Any]]
    args: Dict[str, Any]
    requests: List[RequestRecord]

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable run label."""
        return f'Parallel {self.parallel} / Number {self.number}'

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0-100)."""
        total = self.summary.get('Total requests', 1)
        succeed = self.summary.get('Succeed requests', 0)
        return round(succeed / total * 100, 1) if total > 0 else 0.0

    def get_p99(self, metric_key: str) -> float:
        """Return the P99 value for *metric_key* from the percentile list."""
        for p in self.percentiles:
            if p.get('Percentiles') == '99%':
                return float(p.get(metric_key) or 0)
        return 0.0

    def summary_items(self, is_embedding: bool) -> List[Dict[str, str]]:
        """Return benchmark_summary fields as ``[{'key': ..., 'value': ...}]`` for display."""
        s = self.summary
        rate_raw = s.get('Request rate (req/s)', -1)
        rate_str = 'INF' if rate_raw == -1 else f'{rate_raw:.3f}'

        base = [
            ('Total Requests', str(int(s.get('Total requests', 0)))),
            ('Succeed Requests', str(int(s.get('Succeed requests', 0)))),
            ('Failed Requests', str(int(s.get('Failed requests', 0)))),
            ('Concurrency', str(int(s.get('Number of concurrency', 0)))),
            ('Time Taken (s)', f"{s.get('Time taken for tests (s)', 0):.3f}"),
            ('Request Rate (req/s)', rate_str),
            ('Request Throughput (req/s)', f"{s.get('Request throughput (req/s)', 0):.4f}"),
            ('Avg Latency (s)', f"{s.get('Average latency (s)', 0):.4f}"),
        ]

        if is_embedding:
            extra = [
                ('Input Tok Throughput (tok/s)', f"{s.get('Input token throughput (tok/s)', 0):.2f}"),
                ('Avg Input Tokens', f"{s.get('Average input tokens per request', 0):.1f}"),
            ]
        else:
            extra = [
                ('Output Tok Throughput (tok/s)', f"{s.get('Output token throughput (tok/s)', 0):.2f}"),
                ('Total Tok Throughput (tok/s)', f"{s.get('Total token throughput (tok/s)', 0):.2f}"),
                ('Avg TTFT (s)', f"{s.get('Average time to first token (s)', 0):.4f}"),
                ('Avg TPOT (s)', f"{s.get('Average time per output token (s)', 0):.4f}"),
                ('Avg ITL (s)', f"{s.get('Average inter-token latency (s)', 0):.4f}"),
                ('Avg Input Tokens', f"{s.get('Average input tokens per request', 0):.1f}"),
                ('Avg Output Tokens', f"{s.get('Average output tokens per request', 0):.1f}"),
            ]

        return [{'key': k, 'value': v} for k, v in base + extra]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class RunLoader:
    """Discovers and loads all benchmark runs from an output directory."""

    @staticmethod
    def load_all(output_dir: str) -> List[RunData]:
        """Walk *output_dir* for ``parallel_*`` subdirectories and return sorted runs."""
        if not os.path.isdir(output_dir):
            return []

        runs: List[RunData] = []
        for entry in sorted(os.listdir(output_dir)):
            run_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(run_dir) or not entry.startswith('parallel_'):
                continue
            run = RunLoader._load_single(run_dir, entry)
            if run is not None:
                runs.append(run)

        return sorted(runs, key=lambda r: (r.parallel, r.number))

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_single(run_dir: str, dir_name: str) -> Optional[RunData]:
        summary = RunLoader._load_json(os.path.join(run_dir, 'benchmark_summary.json')) or {}
        percentiles = RunLoader._load_json(os.path.join(run_dir, 'benchmark_percentile.json')) or []
        args = RunLoader._load_json(os.path.join(run_dir, 'benchmark_args.json')) or {}
        requests = RunLoader._load_db(run_dir)

        m = re.match(r'parallel_(\d+)_number_(\d+)', dir_name)
        if m:
            parallel, number = int(m.group(1)), int(m.group(2))
        else:
            parallel = int(summary.get('Number of concurrency', 0))
            number = int(summary.get('Total requests', 0))

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
    def _load_json(path: str) -> Optional[Any]:
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

                    records.append(
                        RequestRecord(
                            start_time=float(d.get('start_time') or 0),
                            completed_time=float(d.get('completed_time') or 0),
                            latency=float(d.get('latency') or 0),
                            first_chunk_latency=d.get('first_chunk_latency'),
                            prompt_tokens=int(d.get('prompt_tokens') or 0),
                            completion_tokens=int(d.get('completion_tokens') or 0),
                            inter_token_latencies=itl,
                            time_per_output_token=d.get('time_per_output_token'),
                            success=bool(d.get('success', 0)),
                        )
                    )
        except Exception as exc:
            logger.warning(f'Failed to read DB {db_path}: {exc}')

        return records
