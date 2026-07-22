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
from typing import List, Optional, Tuple

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
_RATE_RE = re.compile(r'^rate_(-?[\d.]+)_number_(\d+)$')


class RunLoader:
    """Discovers and loads all benchmark runs from an output directory.

    Recognized sub-directory patterns:

    * ``parallel_<N>_number_<M>`` – closed-loop / fixed-concurrency mode
    * ``rate_<R>_number_<M>``     – open-loop / fixed-rate mode
    """

    @staticmethod
    def load_all(output_dir: str, *, with_requests: bool = True) -> List[RunData]:
        """Walk *output_dir* for run subdirectories and return sorted runs.

        Set *with_requests* to ``False`` to skip reading the per-request SQLite
        DB (``benchmark_data.db``). Summary/percentile metadata is cheap to load,
        so list/detail endpoints that only need aggregates should pass ``False``
        to avoid pulling every historical request row into memory.
        """
        if not os.path.isdir(output_dir):
            return []

        runs: List[RunData] = []
        for entry in sorted(os.listdir(output_dir)):
            run_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            if not (_PARALLEL_RE.match(entry) or _RATE_RE.match(entry)):
                continue
            run = RunLoader._load_single(run_dir, entry, with_requests=with_requests)
            if run is not None:
                runs.append(run)

        return sorted(runs, key=lambda r: (r.sort_key, r.number))

    @staticmethod
    def load_run(run_dir: str, dir_name: str, *, with_requests: bool = True) -> Optional[RunData]:
        """Load a single run sub-directory (``parallel_*``/``rate_*``).

        Useful when only one run's per-request data is needed (e.g. a single
        per-request chart) so the sibling runs' DBs stay untouched.
        """
        if not (_PARALLEL_RE.match(dir_name) or _RATE_RE.match(dir_name)):
            return None
        return RunLoader._load_single(run_dir, dir_name, with_requests=with_requests)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_single(run_dir: str, dir_name: str, *, with_requests: bool = True) -> Optional[RunData]:
        summary_dict = RunLoader._load_json(os.path.join(run_dir, 'benchmark_summary.json')) or {}
        percentile_rows = RunLoader._load_json(os.path.join(run_dir, 'benchmark_percentile.json')) or []
        args = RunLoader._load_json(os.path.join(run_dir, 'benchmark_args.json')) or {}
        requests = RunLoader._load_db(run_dir) if with_requests else []

        summary = BenchmarkSummary.from_dict(summary_dict)
        percentiles = PercentileResult.from_list(percentile_rows) if isinstance(percentile_rows, list) \
            else PercentileResult.from_transposed(percentile_rows)

        m_parallel = _PARALLEL_RE.match(dir_name)
        m_rate = _RATE_RE.match(dir_name)
        if m_parallel:
            parallel = int(m_parallel.group(1))
            number = int(m_parallel.group(2))
            rate = None
        elif m_rate:
            # open-loop mode: parallel has no meaningful value; use 0 as a
            # placeholder.  The real ordering key is the rate field.
            parallel = 0
            number = int(m_rate.group(2))
            rate = float(m_rate.group(1))
        else:
            parallel = summary.concurrency
            number = summary.total_requests
            rate = None

        return RunData(
            dir_name=dir_name,
            parallel=parallel,
            number=number,
            rate=rate,
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
    def _row_to_record(d: dict) -> RequestRecord:
        """Convert a raw ``result`` DB row dict into a :class:`RequestRecord`.

        The DB stores all latency fields in seconds; they are converted to ms
        here so callers always see millisecond values.
        """
        itl_raw = d.get('inter_token_latencies')
        try:
            itl: List[float] = json.loads(itl_raw) if isinstance(itl_raw, str) else []
        except (json.JSONDecodeError, TypeError):
            itl = []

        fcl = d.get('first_chunk_latency')
        tpot = d.get('time_per_output_token')
        return RequestRecord(
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

    # Columns selected for a per-request record (shared by full + paged reads).
    _RESULT_COLUMNS = (
        'SELECT start_time, completed_time, latency, first_chunk_latency, '
        'prompt_tokens, completion_tokens, inter_token_latencies, '
        'time_per_output_token, success FROM result'
    )

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
                cursor.execute(RunLoader._RESULT_COLUMNS)
                for row in cursor.fetchall():
                    records.append(RunLoader._row_to_record(dict(row)))
        except Exception as exc:
            logger.warning(f'Failed to read DB {db_path}: {exc}')

        return records

    @staticmethod
    def _status_filter(status: Optional[str]) -> str:
        """Return the SQL WHERE clause for a 'success'|'failed'|None *status*."""
        if status == 'success':
            return ' WHERE success = 1'
        if status == 'failed':
            return ' WHERE success = 0'
        return ''

    @staticmethod
    def count_requests(run_dir: str) -> int:
        """Return the number of per-request rows in ``benchmark_data.db`` (0 if none).

        Runs a ``SELECT COUNT(*)`` so no request rows are materialised in memory.
        """
        db_path = os.path.join(run_dir, 'benchmark_data.db')
        if not os.path.exists(db_path):
            return 0
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM result')
                row = cursor.fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:
            logger.warning(f'Failed to count DB {db_path}: {exc}')
            return 0

    @staticmethod
    def query_requests(
        run_dir: str,
        *,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[RequestRecord], int]:
        """Return a page of request records plus the total matching count.

        Uses SQL ``LIMIT``/``OFFSET`` (ordered by ``rowid``) and a ``COUNT(*)`` so
        only the requested page is read into memory instead of the whole table.
        *status* may be ``'success'``, ``'failed'`` or ``None`` (no filter).
        """
        db_path = os.path.join(run_dir, 'benchmark_data.db')
        if not os.path.exists(db_path):
            return [], 0

        where = RunLoader._status_filter(status)
        records: List[RequestRecord] = []
        total = 0
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f'SELECT COUNT(*) FROM result{where}')
                count_row = cursor.fetchone()
                total = int(count_row[0]) if count_row else 0
                cursor.execute(
                    f'{RunLoader._RESULT_COLUMNS}{where} ORDER BY rowid LIMIT ? OFFSET ?',
                    (limit, offset),
                )
                for row in cursor.fetchall():
                    records.append(RunLoader._row_to_record(dict(row)))
        except Exception as exc:
            logger.warning(f'Failed to query DB {db_path}: {exc}')
            return [], 0

        return records, total
