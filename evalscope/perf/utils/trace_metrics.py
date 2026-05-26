"""Per-trace metrics aggregator for multi-turn benchmarks.

In multi-turn / trace-replay workloads each conversation (one ``trace_id``)
contains multiple sequential turns.  Single-request metrics
(:class:`~evalscope.perf.utils.benchmark_util.MetricsAccumulator`) treat every
turn as an independent sample, which conflates cold prefill (turn 1) with
warm prefix-cache hits (turn N) and hides genuinely trace-shaped quantities
such as time-to-final-answer-token (TTFAT) and eligible cache hit rate.

This module provides:

* :class:`TraceAccumulator` - feed it ``BenchmarkData`` items keyed by
  ``trace_id`` and it builds per-trace turn lists.  Single-turn benchmarks
  (``trace_id is None``) contribute nothing.
* :class:`TraceLevelSummary` - immutable Pydantic snapshot with one row per
  trace-level metric and percentile columns matching the upstream ``trie``
  reporting format (mean / min / p50 / p90 / p95 / p99 / max).

Metric definitions (per trace):

* ``Latency (s)``: ``last_turn.completed_time - first_turn.start_time``.
* ``First-Turn TTFT (s)``: TTFT of the first turn (cold prefill).
* ``TTFAT (s)``: ``last_turn.start_time + last_turn.ttft - first_turn.start_time``
  - end-to-end wall-clock from trace start to the first token of the final
  assistant reply.
* ``Decode TPS``: arithmetic mean over turns of
  ``(completion_tokens - 1) / (latency - ttft)`` for turns with
  ``completion_tokens > 1`` and ``latency > ttft``.  Matches ``trie``'s
  per-trace aggregation.
* ``Cache Hit Rate (%)``: ``sum(cached_tokens) / sum(prompt_tokens)`` over
  turns with ``prompt_tokens > 0``.
* ``Eligible Cache Hit Rate (%)``: ``sum(cached_tokens) / sum(eligible)``
  where ``eligible_i = prev_prompt_tokens + prev_completion_tokens``
  (turn 1 contributes 0 to both numerator and denominator).  Reflects the
  fraction of theoretically-cacheable tokens the server actually reused.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
from tabulate import tabulate
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from evalscope.perf.utils.benchmark_util import BenchmarkData

# ---------------------------------------------------------------------------
# Internal per-trace state
# ---------------------------------------------------------------------------


@dataclass
class _TraceTurn:
    """Single turn within one trace, holding fields needed for derivation."""

    start_time: float
    completed_time: float
    ttft: float
    latency: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    eligible_prompt_tokens: int


@dataclass
class _TraceState:
    trace_id: str
    turns: List[_TraceTurn] = field(default_factory=list)

    # --- Derived per-trace metrics --------------------------------------
    @property
    def latency(self) -> float:
        if not self.turns:
            return 0.0
        return max(self.turns[-1].completed_time - self.turns[0].start_time, 0.0)

    @property
    def first_turn_ttft(self) -> float:
        return self.turns[0].ttft if self.turns else 0.0

    @property
    def ttfat(self) -> float:
        if not self.turns:
            return 0.0
        last = self.turns[-1]
        first = self.turns[0]
        return max((last.start_time + last.ttft) - first.start_time, 0.0)

    @property
    def decode_tps(self) -> float:
        samples: List[float] = []
        for t in self.turns:
            if t.completion_tokens > 1 and t.latency > t.ttft:
                samples.append((t.completion_tokens - 1) / (t.latency - t.ttft))
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    @property
    def cache_hit_rate(self) -> float:
        total_prompt = sum(t.prompt_tokens for t in self.turns)
        if total_prompt <= 0:
            return 0.0
        total_cached = sum(t.cached_tokens for t in self.turns)
        return total_cached / total_prompt * 100.0

    @property
    def eligible_cache_hit_rate(self) -> float:
        eligible_turns = [t for t in self.turns if t.eligible_prompt_tokens > 0]
        if not eligible_turns:
            return 0.0
        total_eligible = sum(t.eligible_prompt_tokens for t in eligible_turns)
        total_cached = sum(t.cached_tokens for t in eligible_turns)
        if total_eligible <= 0:
            return 0.0
        return total_cached / total_eligible * 100.0


# ---------------------------------------------------------------------------
# Public accumulator
# ---------------------------------------------------------------------------


@dataclass
class TraceAccumulator:
    """Builds per-trace turn lists by feeding completed ``BenchmarkData`` items.

    Use :meth:`feed` once per item and :meth:`to_summary` at the end of the run
    to obtain an immutable :class:`TraceLevelSummary`.  Warmup, failed, and
    single-turn (no ``trace_id``) items are silently skipped so this can be
    driven from the same loop as :class:`MetricsAccumulator`.
    """

    _traces: Dict[str, _TraceState] = field(default_factory=dict)

    def feed(self, data: 'BenchmarkData') -> None:
        if data.is_warmup or not data.success or data.trace_id is None:
            return

        state = self._traces.setdefault(data.trace_id, _TraceState(trace_id=data.trace_id))
        if state.turns:
            prev = state.turns[-1]
            eligible = (prev.prompt_tokens or 0) + (prev.completion_tokens or 0)
        else:
            eligible = 0
        state.turns.append(
            _TraceTurn(
                start_time=data.start_time,
                completed_time=data.completed_time,
                ttft=data.first_chunk_latency,
                latency=data.query_latency,
                prompt_tokens=data.prompt_tokens or 0,
                completion_tokens=data.completion_tokens or 0,
                cached_tokens=data.cached_tokens or 0,
                eligible_prompt_tokens=eligible,
            )
        )

    @property
    def n_traces(self) -> int:
        return len(self._traces)

    def to_summary(self) -> 'TraceLevelSummary':
        if not self._traces:
            return TraceLevelSummary(n_traces=0, rows=[])

        traces = list(self._traces.values())
        # Per-metric value vectors (one entry per trace).
        per_metric: List[tuple] = [
            ('Latency (s)', [t.latency for t in traces]),
            ('First-Turn TTFT (s)', [t.first_turn_ttft for t in traces]),
            ('TTFAT (s)', [t.ttfat for t in traces]),
            ('Decode TPS', [t.decode_tps for t in traces]),
            ('Cache Hit Rate (%)', [t.cache_hit_rate for t in traces]),
            ('Eligible Cache Hit Rate (%)', [t.eligible_cache_hit_rate for t in traces]),
        ]

        rows = [TraceMetricStats.from_values(name, values) for name, values in per_metric]
        return TraceLevelSummary(n_traces=len(traces), rows=rows)


# ---------------------------------------------------------------------------
# Pydantic snapshot models
# ---------------------------------------------------------------------------


class TraceMetricStats(BaseModel):
    """Aggregate stats for one trace-level metric across all traces.

    Percentile labels match the upstream ``trie`` per-trace report
    (mean/min/p50/p90/p95/p99/max).
    """

    model_config = ConfigDict(populate_by_name=True)

    metric: str
    mean: float = 0.0
    min: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0

    @classmethod
    def from_values(cls, metric: str, values: List[float]) -> 'TraceMetricStats':
        if not values:
            return cls(metric=metric)
        arr = np.asarray(values, dtype=float)
        return cls(
            metric=metric,
            mean=float(arr.mean()),
            min=float(arr.min()),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            max=float(arr.max()),
        )


class TraceLevelSummary(BaseModel):
    """Per-trace summary table with one row per metric, columns per statistic."""

    n_traces: int = 0
    rows: List[TraceMetricStats] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return self.n_traces == 0 or not self.rows

    def to_dict(self) -> dict:
        return {'n_traces': self.n_traces, 'rows': [r.model_dump() for r in self.rows]}

    def to_table(self) -> str:
        if self.is_empty():
            return ''
        headers = ['Metric', 'mean', 'min', 'p50', 'p90', 'p95', 'p99', 'max']
        body = [[
            r.metric,
            f'{r.mean:.2f}',
            f'{r.min:.2f}',
            f'{r.p50:.2f}',
            f'{r.p90:.2f}',
            f'{r.p95:.2f}',
            f'{r.p99:.2f}',
            f'{r.max:.2f}',
        ] for r in self.rows]
        col_align = ('left', ) + ('right', ) * 7
        return tabulate(
            body,
            headers=headers,
            tablefmt='simple_outline',
            disable_numparse=True,
            colalign=col_align,
        )
