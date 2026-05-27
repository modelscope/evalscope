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
from pydantic import BaseModel, Field
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


# Zero-anchor used for the "Overall" window: makes overall_rates share the
# same code path as the windowed rates without a special-case branch.
_ORIGIN = _Point(t=0.0, cum_completion=0, cum_new_prompt=0, cum_cached_prompt=0)

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

        # Lock wall_start on the first feed so already-appended points keep
        # their ``t`` value stable.  Lowering wall_start later would silently
        # invalidate every prior point's offset (they don't get rewritten).
        # Sub-second start_time jitter between workers is negligible at the
        # window sizes we report on (30s tail, 20% steady-state).
        if self._wall_start is None:
            self._wall_start = data.start_time

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

    def _rates_from(self, anchor: _Point) -> List[float]:
        """Return [total_prompt, new_prompt, cached_prompt, completion] tok/s
        for the window from ``anchor`` to the latest point.

        Returns zeros when the window has no width or no points.
        """
        if not self._points:
            return [0.0, 0.0, 0.0, 0.0]
        last = self._points[-1]
        dt = last.t - anchor.t
        if dt <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        d_new_prompt = last.cum_new_prompt - anchor.cum_new_prompt
        d_cached_prompt = last.cum_cached_prompt - anchor.cum_cached_prompt
        d_completion = last.cum_completion - anchor.cum_completion
        return [
            (d_new_prompt + d_cached_prompt) / dt,
            d_new_prompt / dt,
            d_cached_prompt / dt,
            d_completion / dt,
        ]

    def overall_rates(self) -> List[float]:
        return self._rates_from(_ORIGIN)

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
        # Anchor at the latest point with t <= (wall - window_s) so the window
        # length closely matches window_s while staying on a real sample.
        return self._rates_from(self._find_anchor(wall - window_s))

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
        anchor = self._find_anchor(wall * warmup_frac)
        # If the anchor *is* the last point, the steady-state window has no
        # data; fall back to overall rather than returning zeros.
        if anchor is self._points[-1]:
            return self.overall_rates()
        return self._rates_from(anchor)

    def _find_anchor(self, t_target: float) -> _Point:
        """Return the latest point with ``t <= t_target`` (or the first point)."""
        if not self._points or t_target <= self._points[0].t:
            return self._points[0]
        # Linear forward scan: points are append-only sorted by t (wall_start is
        # locked on first feed), so we can break at the first point past t_target.
        # Timelines are at most a few thousand entries so bisect isn't worth importing.
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
            'points': [{
                't': round(p.t, 6),
                'cum_completion': p.cum_completion,
                'cum_new_prompt': p.cum_new_prompt,
                'cum_cached_prompt': p.cum_cached_prompt,
            } for p in self._points],
        }


# ---------------------------------------------------------------------------
# Pydantic snapshot
# ---------------------------------------------------------------------------


class WorkloadThroughputRow(BaseModel):
    """One throughput row across Overall / Last-window / Steady-state columns."""

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
        return not self.rows

    def to_dict(self) -> dict:
        return self.model_dump()

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
