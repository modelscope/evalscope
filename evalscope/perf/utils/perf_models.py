"""Pydantic v2 data models for the perf benchmark pipeline.

Provides strongly-typed, serialisable classes that replace the bare ``dict``
objects previously produced by ``db_util`` and consumed by ``rich_display``,
``generate_report``, and ``perf_charts``.

Public classes
--------------
    BenchmarkSummary   — one benchmark run's summary metrics
    PercentileRow      — a single percentile row (e.g. P99)
    PercentileResult   — the full percentile table
    RequestRecord      — a single per-request DB row
    RunData            — all data for one parallel_X_number_Y run
"""

from __future__ import annotations

import json
from pydantic import BaseModel, ConfigDict, Field
from tabulate import tabulate
from typing import Any, Dict, List, Optional

from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics

# ---------------------------------------------------------------------------
# BenchmarkSummary
# ---------------------------------------------------------------------------


class BenchmarkSummary(BaseModel):
    """Strongly-typed snapshot of benchmark_summary.json.

    Field names use Python conventions; JSON aliases match the string constants
    in :class:`~evalscope.perf.utils.perf_constants.Metrics` so that
    ``model_dump(by_alias=True)`` produces a dict identical to the legacy
    ``BenchmarkMetrics.create_message()`` output.
    """

    model_config = ConfigDict(populate_by_name=True)

    # --- General ---
    time_taken: float = Field(0.0, alias=Metrics.TIME_TAKEN_FOR_TESTS)
    concurrency: int = Field(0, alias=Metrics.NUMBER_OF_CONCURRENCY)
    request_rate: float = Field(0.0, alias=Metrics.REQUEST_RATE)
    total_requests: int = Field(0, alias=Metrics.TOTAL_REQUESTS)
    succeed_requests: int = Field(0, alias=Metrics.SUCCEED_REQUESTS)
    failed_requests: int = Field(0, alias=Metrics.FAILED_REQUESTS)
    request_throughput: float = Field(0.0, alias=Metrics.REQUEST_THROUGHPUT)
    avg_latency: float = Field(0.0, alias=Metrics.AVERAGE_LATENCY)
    avg_input_tokens: float = Field(0.0, alias=Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST)

    # --- LLM-specific ---
    output_token_throughput: float = Field(0.0, alias=Metrics.OUTPUT_TOKEN_THROUGHPUT)
    total_token_throughput: float = Field(0.0, alias=Metrics.TOTAL_TOKEN_THROUGHPUT)
    avg_ttft: float = Field(0.0, alias=Metrics.AVERAGE_TIME_TO_FIRST_TOKEN)
    avg_tpot: float = Field(0.0, alias=Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN)
    avg_itl: float = Field(0.0, alias=Metrics.AVERAGE_INTER_TOKEN_LATENCY)
    avg_output_tokens: float = Field(0.0, alias=Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST)

    # --- Embedding / Rerank-specific ---
    input_token_throughput: float = Field(0.0, alias=Metrics.INPUT_TOKEN_THROUGHPUT)

    # --- Multi-turn (optional) ---
    avg_turns: Optional[float] = Field(None, alias=Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST)
    avg_cached_percent: Optional[float] = Field(None, alias=Metrics.AVERAGE_CACHED_PERCENT)

    # --- Speculative decoding (optional) ---
    avg_decoded_tokens_per_iter: Optional[float] = Field(None, alias=Metrics.AVERAGE_DECODED_TOKENS_PER_ITER)
    approx_spec_acceptance_rate: Optional[float] = Field(None, alias=Metrics.APPROX_SPECULATIVE_ACCEPTANCE_RATE)

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0–100)."""
        if self.total_requests > 0:
            return round(self.succeed_requests / self.total_requests * 100, 1)
        return 0.0

    # -----------------------------------------------------------------------
    # Factory / serialization helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> 'BenchmarkSummary':
        """Construct from a dict whose keys are the JSON alias strings.

        Tolerant of missing optional fields.
        """
        return cls.model_validate(d)

    def to_dict(self) -> dict:
        """Export as a dict with JSON alias keys (matches benchmark_summary.json)."""
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_table(self) -> str:
        """Render summary metrics as a grouped tabulate table (simple_outline)."""

        def _fmt(v):
            if isinstance(v, float):
                return f'{v:.2f}'
            return str(v)

        rows = []

        # ── General ──
        rows.append(('── General ──', ''))
        rows.append((Metrics.TIME_TAKEN_FOR_TESTS, _fmt(self.time_taken)))
        rows.append((Metrics.NUMBER_OF_CONCURRENCY, str(self.concurrency)))
        rows.append((Metrics.REQUEST_RATE, _fmt(self.request_rate)))
        rows.append(
            ('Total / Success / Failed', f'{self.total_requests} / {self.succeed_requests} / {self.failed_requests}')
        )
        rows.append((Metrics.REQUEST_THROUGHPUT, _fmt(self.request_throughput)))

        # ── Latency ──
        rows.append(('── Latency ──', ''))
        rows.append((Metrics.AVERAGE_LATENCY, _fmt(self.avg_latency)))
        if self.avg_ttft:
            rows.append((Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, _fmt(self.avg_ttft)))
        if self.avg_tpot:
            rows.append((Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, _fmt(self.avg_tpot)))
        if self.avg_itl:
            rows.append((Metrics.AVERAGE_INTER_TOKEN_LATENCY, _fmt(self.avg_itl)))

        # ── Tokens ──
        rows.append(('── Tokens ──', ''))
        rows.append((Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, _fmt(self.avg_input_tokens)))
        if self.avg_output_tokens:
            rows.append((Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST, _fmt(self.avg_output_tokens)))
        if self.output_token_throughput:
            rows.append((Metrics.OUTPUT_TOKEN_THROUGHPUT, _fmt(self.output_token_throughput)))
        if self.total_token_throughput:
            rows.append((Metrics.TOTAL_TOKEN_THROUGHPUT, _fmt(self.total_token_throughput)))
        if self.input_token_throughput:
            rows.append((Metrics.INPUT_TOKEN_THROUGHPUT, _fmt(self.input_token_throughput)))

        # ── Multi-turn (optional) ──
        if self.avg_turns is not None or self.avg_cached_percent is not None:
            rows.append(('── Multi-turn ──', ''))
            if self.avg_turns is not None:
                rows.append((Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST, _fmt(self.avg_turns)))
            if self.avg_cached_percent is not None:
                rows.append((Metrics.AVERAGE_CACHED_PERCENT, _fmt(self.avg_cached_percent)))

        # ── Speculative Decoding (optional) ──
        if self.avg_decoded_tokens_per_iter is not None or self.approx_spec_acceptance_rate is not None:
            rows.append(('── Speculative Decoding ──', ''))
            if self.avg_decoded_tokens_per_iter is not None:
                rows.append((Metrics.AVERAGE_DECODED_TOKENS_PER_ITER, _fmt(self.avg_decoded_tokens_per_iter)))
            if self.approx_spec_acceptance_rate is not None:
                rows.append((Metrics.APPROX_SPECULATIVE_ACCEPTANCE_RATE, _fmt(self.approx_spec_acceptance_rate)))

        raw = tabulate(rows, headers=['Metric', 'Value'], tablefmt='simple_outline', colalign=('left', 'right'))

        return raw


# ---------------------------------------------------------------------------
# PercentileRow / PercentileResult
# ---------------------------------------------------------------------------


class PercentileRow(BaseModel):
    """A single percentile point row (e.g. P50, P99) from benchmark_percentile.json."""

    model_config = ConfigDict(populate_by_name=True)

    percentile: str = Field('', alias=PercentileMetrics.PERCENTILES)
    latency: Optional[float] = Field(None, alias=PercentileMetrics.LATENCY)
    ttft: Optional[float] = Field(None, alias=PercentileMetrics.TTFT)
    itl: Optional[float] = Field(None, alias=PercentileMetrics.ITL)
    tpot: Optional[float] = Field(None, alias=PercentileMetrics.TPOT)
    input_tokens: Optional[float] = Field(None, alias=PercentileMetrics.INPUT_TOKENS)
    output_tokens: Optional[float] = Field(None, alias=PercentileMetrics.OUTPUT_TOKENS)
    output_throughput: Optional[float] = Field(None, alias=PercentileMetrics.OUTPUT_THROUGHPUT)
    input_throughput: Optional[float] = Field(None, alias=PercentileMetrics.INPUT_THROUGHPUT)
    total_throughput: Optional[float] = Field(None, alias=PercentileMetrics.TOTAL_THROUGHPUT)
    decode_throughput: Optional[float] = Field(None, alias=PercentileMetrics.DECODE_THROUGHPUT)

    @classmethod
    def from_dict(cls, d: dict) -> 'PercentileRow':
        """Construct from a dict whose keys are alias strings."""
        return cls.model_validate(d)

    def to_dict(self) -> dict:
        """Export as alias-keyed dict (matches one row of benchmark_percentile.json)."""
        return self.model_dump(by_alias=True, exclude_none=True)


class PercentileResult(BaseModel):
    """The full percentile table, stored as an ordered list of :class:`PercentileRow`."""

    rows: List[PercentileRow] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_transposed(cls, d: dict) -> 'PercentileResult':
        """Construct from the *transposed* dict produced by ``get_percentile_results``.

        Input format::

            {
                'Percentiles': ['10%', '25%', ..., '99%'],
                'Latency (s)': [0.1, 0.2, ..., 0.9],
                'TTFT (s)':    [0.05, ...],
                ...
            }
        """
        if not d:
            return cls(rows=[])
        percentile_labels: List[str] = d.get(PercentileMetrics.PERCENTILES, [])
        n = len(percentile_labels)
        rows = []
        for i in range(n):
            row_dict: Dict[str, Any] = {PercentileMetrics.PERCENTILES: percentile_labels[i]}
            for key, values in d.items():
                if key == PercentileMetrics.PERCENTILES:
                    continue
                if isinstance(values, list) and i < len(values):
                    row_dict[key] = values[i]
            rows.append(PercentileRow.from_dict(row_dict))
        return cls(rows=rows)

    @classmethod
    def from_list(cls, rows_list: List[dict]) -> 'PercentileResult':
        """Construct from a list of row dicts (the stored benchmark_percentile.json format)."""
        if not rows_list:
            return cls(rows=[])
        return cls(rows=[PercentileRow.from_dict(r) for r in rows_list])

    # -----------------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------------

    def get_p(self, percentile_str: str, metric_field: str) -> float:
        """Return the value for *metric_field* at *percentile_str* (e.g. ``'99%'``).

        *metric_field* is the **Python attribute name** on :class:`PercentileRow`
        (e.g. ``'latency'``, ``'ttft'``, ``'tpot'``).
        Returns ``0.0`` if not found.
        """
        for row in self.rows:
            if row.percentile == percentile_str:
                return float(getattr(row, metric_field) or 0)
        return 0.0

    def get_p99(self, metric_field: str) -> float:
        """Return the P99 value for *metric_field*. Convenience wrapper."""
        return self.get_p('99%', metric_field)

    def get_p_by_alias(self, percentile_str: str, alias: str) -> float:
        """Return value by *alias* (the JSON key string) at *percentile_str*.

        Used by legacy code paths that still operate on alias strings.
        Returns ``0.0`` if not found.
        """
        # Build alias -> field name mapping once at call time
        alias_map = {
            field_info.alias: name
            for name, field_info in PercentileRow.model_fields.items()
            if field_info.alias
        }
        field_name = alias_map.get(alias)
        if field_name is None:
            return 0.0
        return self.get_p(percentile_str, field_name)

    def get_p99_by_alias(self, alias: str) -> float:
        """Return P99 value by alias string. Convenience wrapper."""
        return self.get_p_by_alias('99%', alias)

    # -----------------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------------

    def to_columns(self) -> dict:
        """Export as a transposed column dict (matches get_percentile_results output).

        Used by db_util for tabulate display and JSON writing.
        """
        if not self.rows:
            return {}
        result: Dict[str, list] = {PercentileMetrics.PERCENTILES: [r.percentile for r in self.rows]}
        for name, field_info in PercentileRow.model_fields.items():
            alias = field_info.alias
            if alias == PercentileMetrics.PERCENTILES or alias is None:
                continue
            values = [getattr(row, name) for row in self.rows]
            if any(v is not None for v in values):
                result[alias] = [v if v is not None else float('nan') for v in values]
        return result

    def to_list(self) -> List[dict]:
        """Export as a list of alias-keyed row dicts (for benchmark_percentile.json)."""
        return [r.to_dict() for r in self.rows]

    def to_table(self) -> str:
        """Render percentile results as a formatted table string.

        Rows are metrics, columns are percentile labels (e.g. 5%, 10%, ..., 99%).
        All numeric values are formatted to two decimal places.
        """
        col_data = self.to_columns()
        p_labels = col_data.get(PercentileMetrics.PERCENTILES, [])
        rows = [[metric] + [f'{v:.2f}' if isinstance(v, (int, float)) else v
                            for v in values]
                for metric, values in col_data.items()
                if metric != PercentileMetrics.PERCENTILES]
        col_align = ('left', ) + ('right', ) * len(p_labels)
        return tabulate(
            rows,
            headers=['Metric'] + p_labels,
            tablefmt='simple_outline',
            disable_numparse=True,
            colalign=col_align,
        )


# ---------------------------------------------------------------------------
# RequestRecord
# ---------------------------------------------------------------------------


class RequestRecord(BaseModel):
    """A single benchmark request record loaded from the SQLite DB."""

    start_time: float = 0.0
    completed_time: float = 0.0
    latency: float = 0.0
    first_chunk_latency: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inter_token_latencies: List[float] = Field(default_factory=list)
    time_per_output_token: Optional[float] = None
    success: bool = False


# ---------------------------------------------------------------------------
# RunData
# ---------------------------------------------------------------------------


class RunData(BaseModel):
    """All data for a single benchmark run (one parallel_X_number_Y directory)."""

    dir_name: str
    parallel: int
    number: int
    summary: BenchmarkSummary
    percentiles: PercentileResult
    args: Dict[str, Any] = Field(default_factory=dict)
    requests: List[RequestRecord] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable run label.

        * ``parallel_<N>_number_<M>`` directories  → "Parallel N / Number M"
        * ``rate_<R>_number_<M>`` directories       → "Rate R rps / Number M"
        """
        import re
        m = re.match(r'^rate_([\d.]+)_number_(\d+)$', self.dir_name)
        if m:
            return f'Rate {m.group(1)} rps / Number {m.group(2)}'
        return f'Parallel {self.parallel} / Number {self.number}'

    @property
    def success_rate(self) -> float:
        """Success rate delegated to BenchmarkSummary."""
        return self.summary.success_rate

    def get_p99(self, metric_field: str) -> float:
        """P99 lookup delegated to PercentileResult.

        *metric_field* may be either a Python attribute name (e.g. ``'latency'``)
        or a JSON alias string (e.g. ``'Latency (s)'``).
        """
        # Check if it looks like an alias (contains spaces / parentheses)
        if ' ' in metric_field or '(' in metric_field:
            return self.percentiles.get_p99_by_alias(metric_field)
        return self.percentiles.get_p99(metric_field)

    # -----------------------------------------------------------------------
    # JSON helper for serializing whole RunData (optional convenience)
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Export the run summary and percentiles as plain dicts."""
        return {
            'dir_name': self.dir_name,
            'parallel': self.parallel,
            'number': self.number,
            'summary': self.summary.to_dict(),
            'percentiles': self.percentiles.to_list(),
            'args': self.args,
        }
