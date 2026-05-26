"""Cumulative token timeline + workload-level throughput aggregator.

Mirrors trie's ``CompletionPoint`` / ``ServerMetrics`` reporting so multi-turn
runs can report throughput under three windows that single-request averages
would smear together:

* ``Overall``        - total tokens / total wall-clock time
* ``Last 30s``       - sliding tail-window, captures end-of-run behaviour
* ``Steady-state``   - drops the first ``warmup_frac`` (default 20%) of
  wall-clock to exclude the ramp where the server is still spinning up its
  KV cache, request scheduler, and (for batched engines) reaching steady
  batch sizes.  Matches trie's steady-state definition.

Timeline points are kept in memory only (raw points may also be exported
to ``workload_timeline.json`` so downstream tools / notebooks can re-derive
custom windows with pandas).  The SQLite layer is intentionally untouched.

Per-point fields:

* ``t``                  - seconds since wall-clock start (first request's
  ``start_time``)
* ``cum_completion``     - cumulative completion tokens
* ``cum_new_prompt``     - cumulative ``prompt_tokens - cached_tokens``
  (the part actually computed; ``cached_tokens`` defaults to 0 when the
  server doesn't report it, so single-turn runs degenerate to
  ``cum_total_prompt``)
* ``cum_cached_prompt``  - cumulative ``cached_tokens``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
from tabulate import tabulate
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from evalscope.perf.utils.benchmark_util import BenchmarkData


# ---------------------------------------------------------------------------
# Internal point
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    t: float
    cum_completion: int
    cum_new_prompt: int
    cum_cached_prompt: int


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class WorkloadTimeline:
    """Append-only timeline of cumulative tokens, keyed off completion time.

    Feed every successful, non-warmup :class:`BenchmarkData` via :meth:`feed`;
    failed / warmup items are silently skipped.  Call :meth:`to_summary` at the
    end of the run to obtain a :class:`WorkloadThroughput` snapshot.
    """

    _wall_start: Optional[float] = None
    _points: List[_Point] = field(default_factory=list)

    # Running cumulative totals (kept outside _points so feed() is O(1)).
    _cum_completion: int = 0
    _cum_new_prompt: int = 0
    _cum_cached_prompt: int = 0

    def feed(self, data: 'BenchmarkData') -> None:
        if data.is_warmup or not data.success:
            return
        # Drop items missing either timestamp or where completion is non-monotonic;
        # both indicate the record was never actually populated by the HTTP layer.
        if data.completed_time <= 0 or data.completed_time < data.start_time:
            return

        if self._wall_start is None:
            self._wall_start = data.start_time
        else:
            self._wall_start = min(self._wall_start, data.start_time)

        prompt = data.prompt_tokens or 0
        completion = data.completion_tokens or 0
        cached = data.cached_tokens or 0
        # Server-side new-prompt cost.  Clamp to >=0 because some servers can
        # report cached > prompt when the chat template inflates the prompt
        # post-tokenization (rare but observed on a few OpenAI-compat backends).
        new_prompt = max(prompt - cached, 0)

        self._cum_completion += completion
        self._cum_new_prompt += new_prompt
        self._cum_cached_prompt += cached

        self._points.append(
            _Point(
                t=data.completed_time - self._wall_start,
                cum_completion=self._cum_completion,
                cum_new_prompt=self._cum_new_prompt,
                cum_cached_prompt=self._cum_cached_prompt,
            )
        )

    # ------------------------------------------------------------------
    # Derived rates
    # ------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self._points)

    @property
    def wall_time(self) -> float:
        """End-to-end wall-clock duration (first start -> last completion)."""
        if not self._points:
            return 0.0
        return max(self._points[-1].t, 0.0)

    def _rates(self, *, t_start: float, t_end: float, cum_completion_start: int,
               cum_new_prompt_start: int, cum_cached_prompt_start: int) -> List[float]:
        """Return [total_prompt, new_prompt, cached_prompt, completion] tok/s
        for a window bounded by absolute timestamps and starting cumulative counts.

        Returns zeros when the window has no width or no points.
        """
        if not self._points or t_end <= t_start:
            return [0.0, 0.0, 0.0, 0.0]
        last = self._points[-1]
        dt = t_end - t_start
        d_new_prompt = last.cum_new_prompt - cum_new_prompt_start
        d_cached_prompt = last.cum_cached_prompt - cum_cached_prompt_start
        d_completion = last.cum_completion - cum_completion_start
        return [
            (d_new_prompt + d_cached_prompt) / dt,
            d_new_prompt / dt,
            d_cached_prompt / dt,
            d_completion / dt,
        ]

    def overall_rates(self) -> List[float]:
        return self._rates(
            t_start=0.0,
            t_end=self.wall_time,
            cum_completion_start=0,
            cum_new_prompt_start=0,
            cum_cached_prompt_start=0,
        )

    def last_window_rates(self, window_s: float) -> List[float]:
        """Sliding tail window of length ``window_s`` seconds.

        If the run is shorter than ``window_s`` we fall back to ``overall_rates``
        - quoting trie's "Last 30s" on a 5s run would otherwise be misleading.
        """
        if not self._points or window_s <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        wall = self.wall_time
        if wall <= window_s:
            return self.overall_rates()

        t_start = wall - window_s
        # Find the cumulative counts at or just before t_start to compute deltas.
        anchor = self._find_anchor(t_start)
        return self._rates(
            t_start=anchor.t,
            t_end=self._points[-1].t,
            cum_completion_start=anchor.cum_completion,
            cum_new_prompt_start=anchor.cum_new_prompt,
            cum_cached_prompt_start=anchor.cum_cached_prompt,
        )

    def steady_state_rates(self, warmup_frac: float = 0.2) -> List[float]:
        """Drop the first ``warmup_frac`` of wall_time and rate over what remains.

        Matches trie's steady-state window: takes the latest cumulative counts
        and subtracts the counts at the moment ``warmup_frac * wall_time`` elapsed.
        Falls back to ``overall_rates`` when the timeline is too short to
        meaningfully discard a warmup region.
        """
        if not self._points:
            return [0.0, 0.0, 0.0, 0.0]
        wall = self.wall_time
        if wall <= 0 or warmup_frac <= 0:
            return self.overall_rates()

        cutoff_t = wall * warmup_frac
        anchor = self._find_anchor(cutoff_t)
        # If the anchor *is* the last point, the steady-state window has no
        # data; fall back to overall rather than returning zeros.
        if anchor is self._points[-1]:
            return self.overall_rates()
        return self._rates(
            t_start=anchor.t,
            t_end=self._points[-1].t,
            cum_completion_start=anchor.cum_completion,
            cum_new_prompt_start=anchor.cum_new_prompt,
            cum_cached_prompt_start=anchor.cum_cached_prompt,
        )

    def _find_anchor(self, t_target: float) -> _Point:
        """Return the latest point with ``t <= t_target`` (or the first point)."""
        if not self._points or t_target <= self._points[0].t:
            return self._points[0]
        # Linear scan from the end - timelines are at most a few thousand entries
        # and inserts are append-only so we don't import bisect just for this.
        chosen = self._points[0]
        for p in self._points:
            if p.t <= t_target:
                chosen = p
            else:
                break
        return chosen

    # ------------------------------------------------------------------
    # Snapshot / serialisation
    # ------------------------------------------------------------------

    def to_summary(self, *, last_window_s: float = 30.0, warmup_frac: float = 0.2) -> 'WorkloadThroughput':
        overall = self.overall_rates()
        last = self.last_window_rates(last_window_s)
        steady = self.steady_state_rates(warmup_frac)
        labels = [
            'Total Prompt tok/s',
            'New Prompt tok/s',
            'Cached Prompt tok/s',
            'Completion tok/s',
        ]
        rows = [
            WorkloadThroughputRow(metric=label, overall=overall[i], last_window=last[i], steady_state=steady[i])
            for i, label in enumerate(labels)
        ]
        return WorkloadThroughput(
            n_samples=self.n_points,
            wall_time_s=round(self.wall_time, 4),
            last_window_s=last_window_s,
            warmup_frac=warmup_frac,
            rows=rows,
        )

    def to_raw_points_dict(self) -> dict:
        """Export raw cumulative-token points for downstream pandas / plots."""
        return {
            'wall_start': self._wall_start,
            'points': [
                {
                    't': round(p.t, 6),
                    'cum_completion': p.cum_completion,
                    'cum_new_prompt': p.cum_new_prompt,
                    'cum_cached_prompt': p.cum_cached_prompt,
                } for p in self._points
            ],
        }


# ---------------------------------------------------------------------------
# Pydantic snapshot
# ---------------------------------------------------------------------------


class WorkloadThroughputRow(BaseModel):
    """One throughput row across Overall / Last-window / Steady-state columns."""

    model_config = ConfigDict(populate_by_name=True)

    metric: str
    overall: float = 0.0
    last_window: float = 0.0
    steady_state: float = 0.0


class WorkloadThroughput(BaseModel):
    """Workload-level throughput summary table.

    ``last_window_s`` and ``warmup_frac`` are recorded alongside the values so a
    consumer can interpret the columns without guessing what window was used.
    """

    n_samples: int = 0
    wall_time_s: float = 0.0
    last_window_s: float = 30.0
    warmup_frac: float = 0.2
    rows: List[WorkloadThroughputRow] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return self.n_samples == 0 or not self.rows

    def to_dict(self) -> dict:
        return {
            'n_samples': self.n_samples,
            'wall_time_s': self.wall_time_s,
            'last_window_s': self.last_window_s,
            'warmup_frac': self.warmup_frac,
            'rows': [r.model_dump() for r in self.rows],
        }

    def to_table(self) -> str:
        if self.is_empty():
            return ''
        last_label = f'Last {int(self.last_window_s)}s'
        steady_label = f'Steady (drop {int(self.warmup_frac * 100)}%)'
        headers = ['Metric (tok/s)', 'Overall', last_label, steady_label]
        body = [[
            r.metric,
            f'{r.overall:.2f}',
            f'{r.last_window:.2f}',
            f'{r.steady_state:.2f}',
        ] for r in self.rows]
        col_align = ('left', ) + ('right', ) * 3
        return tabulate(
            body,
            headers=headers,
            tablefmt='simple_outline',
            disable_numparse=True,
            colalign=col_align,
        )
