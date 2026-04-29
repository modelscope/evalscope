"""Plotly-based chart builders for perf benchmark HTML reports.

All public ``build_*`` functions accept ``RunData`` objects (from perf_data)
and return an HTML ``<div>`` string ready for Jinja2 template injection.

Design choices:
  - Every chart uses the ``plotly_dark`` template with transparent backgrounds
    so it blends seamlessly with the dark CSS theme.
  - Bar charts are replaced by line charts (with optional fill) to show trends
    more clearly, as requested.
  - ``ChartBuilder`` is a low-level helper; the public ``build_*`` functions
    are the stable API consumed by generate_report.py.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from evalscope.perf.utils.perf_models import RunData

# ── Design tokens (must match report.css :root variables) ─────────────────────
ACCENT = '#63b3ed'
PURPLE = '#a78bfa'
GREEN = '#34d399'
YELLOW = '#fbbf24'
RED = '#f87171'

PLOTLY_CONFIG: Dict[str, Any] = {'responsive': True}

_GRID = dict(
    gridcolor='rgba(99,179,237,0.08)',
    zerolinecolor='rgba(99,179,237,0.12)',
)

_LAYOUT_BASE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(
        family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        size=13,
    ),
    margin=dict(l=60, r=30, t=40, b=50),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis=dict(**_GRID),
    yaxis=dict(**_GRID),
)

# ---------------------------------------------------------------------------
# ChartBuilder — low-level Plotly helper
# ---------------------------------------------------------------------------


class ChartBuilder:
    """Generic Plotly chart factory shared by all perf chart builders."""

    @staticmethod
    def to_div(fig, div_id: str = '') -> str:
        """Serialise *fig* to an HTML ``<div>`` string (no full page, no CDN)."""
        kwargs: Dict[str, Any] = dict(
            full_html=False,
            include_plotlyjs=False,
            config=PLOTLY_CONFIG,
        )
        if div_id:
            kwargs['div_id'] = div_id
        return fig.to_html(**kwargs)

    @staticmethod
    def layout(**overrides: Any) -> dict:
        """Return a layout dict by merging *_LAYOUT_BASE* with *overrides*."""
        merged = dict(_LAYOUT_BASE)
        merged.update(overrides)
        return merged

    @classmethod
    def line(
        cls,
        traces: list,
        x_title: str,
        y_title: str,
        div_id: str = '',
        extra_layout: Optional[dict] = None,
    ) -> str:
        """Build a multi-trace ``Scatter`` (line) chart and return an HTML div."""
        import plotly.graph_objects as go

        fig = go.Figure()
        for trace in traces:
            fig.add_trace(go.Scatter(**trace))

        layout_kwargs = dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            xaxis=dict(dtick=1, **_GRID),
            yaxis=dict(**_GRID),
        )
        if extra_layout:
            layout_kwargs.update(extra_layout)

        fig.update_layout(**cls.layout(**layout_kwargs))
        return cls.to_div(fig, div_id)

    @classmethod
    def scatter(
        cls,
        traces: list,
        x_title: str,
        y_title: str,
        div_id: str = '',
    ) -> str:
        """Build a ``Scatter`` (marker-only) chart and return an HTML div."""
        import plotly.graph_objects as go

        fig = go.Figure()
        for trace in traces:
            fig.add_trace(go.Scatter(**trace))

        fig.update_layout(
            **cls.layout(
                xaxis_title=x_title,
                yaxis_title=y_title,
                xaxis=dict(**_GRID),
                yaxis=dict(**_GRID),
                showlegend=False,
            )
        )
        return cls.to_div(fig, div_id)

    @classmethod
    def histogram(
        cls,
        x_data: list,
        x_title: str,
        y_title: str,
        div_id: str = '',
    ) -> str:
        """Build a ``Histogram`` chart and return an HTML div."""
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=x_data,
                name='Distribution',
                marker=dict(color=ACCENT, line=dict(color=ACCENT, width=0.5)),
                opacity=0.8,
                nbinsx=50,
            )
        )
        fig.update_layout(
            **cls.layout(
                xaxis_title=x_title,
                yaxis_title=y_title,
                xaxis=dict(**_GRID),
                yaxis=dict(**_GRID),
                bargap=0.05,
            )
        )
        return cls.to_div(fig, div_id)


# ---------------------------------------------------------------------------
# X-axis helper: adapts to open-loop (rate) vs closed-loop (concurrency)
# ---------------------------------------------------------------------------

_RATE_DIR_RE = re.compile(r'^rate_[\d.]+_number_\d+$')


def _x_axis(runs: 'List[RunData]'):
    """Return *(xs, x_title)* appropriate for the run mode.

    * open-loop  (``rate_*_number_*`` dirs) → xs = request_rate floats,
      x_title = ``'Rate (req/s)'``
    * closed-loop (``parallel_*_number_*`` dirs) → xs = concurrency integers,
      x_title = ``'Concurrency'``
    """
    is_open = any(_RATE_DIR_RE.match(r.dir_name) for r in runs)
    if is_open:
        return [r.summary.request_rate for r in runs], 'Rate (req/s)'
    return [r.parallel for r in runs], 'Concurrency'


# ---------------------------------------------------------------------------
# Per-sweep charts (one data point per run)
# ---------------------------------------------------------------------------


def build_latency_chart(runs: List[RunData]) -> str:
    """Line chart: Avg + P99 end-to-end latency vs concurrency / rate."""
    xs, x_title = _x_axis(runs)
    traces = [
        dict(
            x=xs,
            y=[r.summary.avg_latency for r in runs],
            mode='lines+markers',
            name='Avg Latency',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('latency') for r in runs],
            mode='lines+markers',
            name='P99 Latency',
            line=dict(color=PURPLE, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title=x_title, y_title='Latency (s)', div_id='chart-latency')


def build_ttft_chart(runs: List[RunData]) -> str:
    """Line chart: Avg + P99 Time-To-First-Token vs concurrency / rate (LLM only)."""
    xs, x_title = _x_axis(runs)
    traces = [
        dict(
            x=xs,
            y=[r.summary.avg_ttft * 1000 for r in runs],
            mode='lines+markers',
            name='Avg TTFT',
            line=dict(color=GREEN, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('ttft') * 1000 for r in runs],
            mode='lines+markers',
            name='P99 TTFT',
            line=dict(color=YELLOW, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title=x_title, y_title='TTFT (ms)', div_id='chart-ttft')


def build_tpot_chart(runs: List[RunData]) -> str:
    """Line chart: Avg + P99 Time-Per-Output-Token vs concurrency / rate (LLM only)."""
    xs, x_title = _x_axis(runs)
    traces = [
        dict(
            x=xs,
            y=[r.summary.avg_tpot * 1000 for r in runs],
            mode='lines+markers',
            name='Avg TPOT',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('tpot') * 1000 for r in runs],
            mode='lines+markers',
            name='P99 TPOT',
            line=dict(color=RED, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title=x_title, y_title='TPOT (ms)', div_id='chart-tpot')


def build_rps_chart(runs: List[RunData]) -> str:
    """Line chart: request throughput (RPS) vs concurrency / rate."""
    xs, x_title = _x_axis(runs)
    traces = [
        dict(
            x=xs,
            y=[r.summary.request_throughput for r in runs],
            mode='lines+markers',
            name='RPS',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(99,179,237,0.08)',
        ),
    ]
    return ChartBuilder.line(traces, x_title=x_title, y_title='Requests/sec', div_id='chart-rps')


def build_throughput_chart(runs: List[RunData], is_embedding: bool) -> str:
    """Line chart: token throughput vs concurrency / rate."""
    xs, x_title = _x_axis(runs)

    if is_embedding:
        traces = [
            dict(
                x=xs,
                y=[r.summary.input_token_throughput for r in runs],
                mode='lines+markers',
                name='Input tok/s',
                line=dict(color=GREEN, width=2),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(52,211,153,0.08)',
            )
        ]
    else:
        traces = [
            dict(
                x=xs,
                y=[r.summary.output_token_throughput for r in runs],
                mode='lines+markers',
                name='Output tok/s',
                line=dict(color=GREEN, width=2),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(52,211,153,0.08)',
            ),
            dict(
                x=xs,
                y=[r.summary.total_token_throughput for r in runs],
                mode='lines+markers',
                name='Total tok/s',
                line=dict(color=PURPLE, width=2, dash='dash'),
                marker=dict(size=8),
            ),
        ]

    return ChartBuilder.line(
        traces,
        x_title=x_title,
        y_title='Tokens/sec',
        div_id='chart-throughput',
    )


def build_success_chart(runs: List[RunData]) -> str:
    """Line chart: success rate (%) vs concurrency / rate."""
    xs, x_title = _x_axis(runs)
    traces = [
        dict(
            x=xs,
            y=[r.success_rate for r in runs],
            mode='lines+markers',
            name='Success Rate (%)',
            line=dict(color=GREEN, width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(52,211,153,0.08)',
        ),
    ]
    return ChartBuilder.line(
        traces,
        x_title=x_title,
        y_title='Success Rate (%)',
        div_id='chart-success',
        extra_layout=dict(yaxis=dict(range=[0, 105], **_GRID))
    )


# ---------------------------------------------------------------------------
# Per-run detail charts
# ---------------------------------------------------------------------------


def build_request_detail_tabs(run: 'RunData', is_embedding: bool) -> list:
    """Build per-request line charts (sorted by start_time) organised as tabs.

    Tabs returned:
      - Latency      : End-to-end latency in seconds (LLM & embedding)
      - TTFT / TPOT  : TTFT and TPOT in milliseconds (LLM only)
      - Tokens       : Prompt + Completion tokens
      - ITL          : avg inter-token latency per request (LLM only)
      - Success      : per-request success/failure markers
    """
    if not run.requests:
        return []

    sorted_reqs = sorted(run.requests, key=lambda r: r.start_time)
    xs = list(range(len(sorted_reqs)))
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', run.dir_name)

    tabs = []

    # ── Tab 1: Latency (seconds) ──────────────────────────────────────────
    lat_traces = [
        dict(
            x=xs,
            y=[r.latency for r in sorted_reqs],
            mode='lines+markers',
            name='Latency',
            line=dict(color=ACCENT, width=1.5),
            marker=dict(size=4),
        )
    ]
    tabs.append({
        'label': 'Latency',
        'chart': ChartBuilder.line(
            lat_traces,
            x_title='Request Index',
            y_title='Latency (s)',
            div_id=f'req-latency-{safe}',
        ),
    })

    # ── Tab 2: TTFT / TPOT / ITL (milliseconds, LLM only) ───────────────────
    if not is_embedding:
        itl_y = [
            ((sum(r.inter_token_latencies) / len(r.inter_token_latencies)) * 1000) if r.inter_token_latencies else 0
            for r in sorted_reqs
        ]
        ttft_tpot_itl_traces = [
            dict(
                x=xs,
                y=[(r.first_chunk_latency * 1000) if r.first_chunk_latency is not None else 0 for r in sorted_reqs],
                mode='lines+markers',
                name='TTFT',
                line=dict(color=GREEN, width=1.5),
                marker=dict(size=4),
            ),
            dict(
                x=xs,
                y=[(r.time_per_output_token * 1000) if r.time_per_output_token is not None else 0 for r in sorted_reqs],
                mode='lines+markers',
                name='TPOT',
                line=dict(color=YELLOW, width=1.5),
                marker=dict(size=4),
            ),
            dict(
                x=xs,
                y=itl_y,
                mode='lines+markers',
                name='Avg ITL',
                line=dict(color=PURPLE, width=1.5),
                marker=dict(size=4),
            ),
        ]
        tabs.append({
            'label': 'TTFT / TPOT / ITL',
            'chart': ChartBuilder.line(
                ttft_tpot_itl_traces,
                x_title='Request Index',
                y_title='Time (ms)',
                div_id=f'req-ttft-tpot-itl-{safe}',
            ),
        })

    # ── Tab 3: Tokens ─────────────────────────────────────────────────────
    tok_traces = [
        dict(
            x=xs,
            y=[r.prompt_tokens for r in sorted_reqs],
            mode='lines+markers',
            name='Prompt Tokens',
            line=dict(color=PURPLE, width=1.5),
            marker=dict(size=4),
        ),
    ]
    if not is_embedding:
        tok_traces.append(
            dict(
                x=xs,
                y=[r.completion_tokens for r in sorted_reqs],
                mode='lines+markers',
                name='Completion Tokens',
                line=dict(color=GREEN, width=1.5),
                marker=dict(size=4),
            )
        )
    tabs.append({
        'label': 'Tokens',
        'chart': ChartBuilder.line(
            tok_traces,
            x_title='Request Index',
            y_title='Token Count',
            div_id=f'req-tokens-{safe}',
        ),
    })

    # ── Tab 4: Success ────────────────────────────────────────────────────
    tabs.append({
        'label': 'Success',
        'chart': ChartBuilder.line(
            [
                dict(
                    x=xs,
                    y=[1 if r.success else 0 for r in sorted_reqs],
                    mode='lines+markers',
                    name='Success',
                    line=dict(color=GREEN, width=1.5),
                    marker=dict(
                        color=[GREEN if r.success else RED for r in sorted_reqs],
                        size=6,
                        symbol=['circle' if r.success else 'x' for r in sorted_reqs],
                    ),
                    hovertemplate='Request %{x}<br>%{customdata}<extra></extra>',
                    customdata=['OK' if r.success else 'FAIL' for r in sorted_reqs],
                )
            ],
            x_title='Request Index',
            y_title='Success (1 = OK, 0 = Fail)',
            div_id=f'req-success-{safe}',
            extra_layout=dict(yaxis=dict(range=[-0.2, 1.4], tickvals=[0, 1], ticktext=['Fail', 'OK'], **_GRID)),
        ),
    })

    return tabs


def build_percentile_chart(run: 'RunData', is_embedding: bool) -> 'Tuple[str, str]':
    """Line charts: metric values across percentile levels (P10-P99).

    Returns a tuple ``(latency_chart_html, token_latency_chart_html)`` where:
      - ``latency_chart_html``       : Latency in seconds.
      - ``token_latency_chart_html`` : TTFT / TPOT / ITL in milliseconds
                                       (empty string for embedding models).
    """
    if not run.percentiles.rows:
        return '', ''

    xs = [row.percentile for row in run.percentiles.rows]
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', run.dir_name)

    # Chart 1: Latency (s)
    latency_traces = [
        dict(
            x=xs,
            y=[row.latency or 0 for row in run.percentiles.rows],
            mode='lines+markers',
            name='Latency',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=6),
        )
    ]
    latency_chart = ChartBuilder.line(
        latency_traces,
        x_title='Percentile',
        y_title='Latency (s)',
        div_id=f'chart-percentile-latency-{safe}',
    )

    # Chart 2: TTFT / TPOT / ITL (ms) — LLM only
    token_lat_chart = ''
    if not is_embedding:
        token_lat_traces = [
            dict(
                x=xs,
                y=[(row.ttft or 0) * 1000 for row in run.percentiles.rows],
                mode='lines+markers',
                name='TTFT',
                line=dict(color=GREEN, width=2),
                marker=dict(size=6),
            ),
            dict(
                x=xs,
                y=[(row.tpot or 0) * 1000 for row in run.percentiles.rows],
                mode='lines+markers',
                name='TPOT',
                line=dict(color=YELLOW, width=2),
                marker=dict(size=6),
            ),
            dict(
                x=xs,
                y=[(row.itl or 0) * 1000 for row in run.percentiles.rows],
                mode='lines+markers',
                name='ITL',
                line=dict(color=PURPLE, width=2),
                marker=dict(size=6),
            ),
        ]
        token_lat_chart = ChartBuilder.line(
            token_lat_traces,
            x_title='Percentile',
            y_title='TTFT / TPOT / ITL (ms)',
            div_id=f'chart-percentile-token-lat-{safe}',
        )

    return latency_chart, token_lat_chart
