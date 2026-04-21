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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .perf_data import RunData

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
# Concurrency-level charts (one data point per run)
# ---------------------------------------------------------------------------


def build_latency_chart(runs: List[RunData]) -> str:
    """Line chart: Avg + P99 end-to-end latency vs concurrency."""
    xs = [r.parallel for r in runs]
    traces = [
        dict(
            x=xs,
            y=[r.summary.get('Average latency (s)', 0) for r in runs],
            mode='lines+markers',
            name='Avg Latency',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('Latency (s)') for r in runs],
            mode='lines+markers',
            name='P99 Latency',
            line=dict(color=PURPLE, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title='Concurrency', y_title='Latency (s)', div_id='chart-latency')


def build_ttft_chart(runs: List[RunData]) -> str:
    """Line chart: Avg + P99 Time-To-First-Token vs concurrency (LLM only)."""
    xs = [r.parallel for r in runs]
    traces = [
        dict(
            x=xs,
            y=[r.summary.get('Average time to first token (s)', 0) for r in runs],
            mode='lines+markers',
            name='Avg TTFT',
            line=dict(color=GREEN, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('TTFT (s)') for r in runs],
            mode='lines+markers',
            name='P99 TTFT',
            line=dict(color=YELLOW, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title='Concurrency', y_title='TTFT (s)', div_id='chart-ttft')


def build_tpot_chart(runs: List[RunData]) -> str:
    """Line chart: Avg + P99 Time-Per-Output-Token vs concurrency (LLM only)."""
    xs = [r.parallel for r in runs]
    traces = [
        dict(
            x=xs,
            y=[r.summary.get('Average time per output token (s)', 0) for r in runs],
            mode='lines+markers',
            name='Avg TPOT',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('TPOT (s)') for r in runs],
            mode='lines+markers',
            name='P99 TPOT',
            line=dict(color=RED, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title='Concurrency', y_title='TPOT (s)', div_id='chart-tpot')


def build_rps_chart(runs: List[RunData]) -> str:
    """Line chart: request throughput (RPS) vs concurrency."""
    xs = [r.parallel for r in runs]
    traces = [
        dict(
            x=xs,
            y=[r.summary.get('Request throughput (req/s)', 0) for r in runs],
            mode='lines+markers',
            name='RPS',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(99,179,237,0.08)',
        ),
    ]
    return ChartBuilder.line(traces, x_title='Concurrency', y_title='Requests/sec', div_id='chart-rps')


def build_throughput_chart(runs: List[RunData], is_embedding: bool) -> str:
    """Line chart: token throughput vs concurrency."""
    xs = [r.parallel for r in runs]

    if is_embedding:
        traces = [
            dict(
                x=xs,
                y=[r.summary.get('Input token throughput (tok/s)', 0) for r in runs],
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
                y=[r.summary.get('Output token throughput (tok/s)', 0) for r in runs],
                mode='lines+markers',
                name='Output tok/s',
                line=dict(color=GREEN, width=2),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(52,211,153,0.08)',
            ),
            dict(
                x=xs,
                y=[r.summary.get('Total token throughput (tok/s)', 0) for r in runs],
                mode='lines+markers',
                name='Total tok/s',
                line=dict(color=PURPLE, width=2, dash='dash'),
                marker=dict(size=8),
            ),
        ]

    return ChartBuilder.line(
        traces,
        x_title='Concurrency',
        y_title='Tokens/sec',
        div_id='chart-throughput',
    )


def build_success_chart(runs: List[RunData]) -> str:
    """Line chart: success rate (%) vs concurrency."""
    xs = [r.parallel for r in runs]
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
        x_title='Concurrency',
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
      - Latency  : Latency + TTFT + TPOT (LLM) / Latency only (embedding)
      - Tokens   : Prompt + Completion tokens
      - ITL      : avg inter-token latency per request (LLM only)
      - Success  : per-request success/failure markers
    """
    if not run.requests:
        return []

    sorted_reqs = sorted(run.requests, key=lambda r: r.start_time)
    xs = list(range(len(sorted_reqs)))
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', run.dir_name)

    tabs = []

    # ── Tab 1: Latency ────────────────────────────────────────────────────
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
    if not is_embedding:
        lat_traces.append(
            dict(
                x=xs,
                y=[r.first_chunk_latency if r.first_chunk_latency is not None else 0 for r in sorted_reqs],
                mode='lines+markers',
                name='TTFT',
                line=dict(color=GREEN, width=1.5),
                marker=dict(size=4),
            )
        )
        lat_traces.append(
            dict(
                x=xs,
                y=[r.time_per_output_token if r.time_per_output_token is not None else 0 for r in sorted_reqs],
                mode='lines+markers',
                name='TPOT',
                line=dict(color=YELLOW, width=1.5),
                marker=dict(size=4),
            )
        )
    tabs.append({
        'label': 'Latency',
        'chart': ChartBuilder.line(
            lat_traces,
            x_title='Request Index',
            y_title='Time (s)',
            div_id=f'req-latency-{safe}',
        ),
    })

    # ── Tab 2: Tokens ─────────────────────────────────────────────────────
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

    # ── Tab 3: ITL (LLM only) ─────────────────────────────────────────────
    if not is_embedding:
        itl_y = [(sum(r.inter_token_latencies) / len(r.inter_token_latencies)) if r.inter_token_latencies else 0
                 for r in sorted_reqs]
        tabs.append({
            'label': 'ITL',
            'chart': ChartBuilder.line(
                [
                    dict(
                        x=xs,
                        y=itl_y,
                        mode='lines+markers',
                        name='Avg ITL',
                        line=dict(color=YELLOW, width=1.5),
                        marker=dict(size=4),
                    )
                ],
                x_title='Request Index',
                y_title='Avg ITL (s)',
                div_id=f'req-itl-{safe}',
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


def build_percentile_chart(run: RunData, is_embedding: bool) -> str:
    """Line chart: metric values across percentile levels (P10-P99)."""
    if not run.percentiles:
        return ''

    xs = [p.get('Percentiles', '') for p in run.percentiles]
    traces = [
        dict(
            x=xs,
            y=[p.get('Latency (s)', 0) for p in run.percentiles],
            mode='lines+markers',
            name='Latency',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=6),
        )
    ]

    if not is_embedding:
        traces += [
            dict(
                x=xs,
                y=[p.get('TTFT (s)', 0) for p in run.percentiles],
                mode='lines+markers',
                name='TTFT',
                line=dict(color=GREEN, width=2),
                marker=dict(size=6),
            ),
            dict(
                x=xs,
                y=[p.get('TPOT (s)', 0) for p in run.percentiles],
                mode='lines+markers',
                name='TPOT',
                line=dict(color=YELLOW, width=2),
                marker=dict(size=6),
            ),
            dict(
                x=xs,
                y=[p.get('ITL (s)', 0) for p in run.percentiles],
                mode='lines+markers',
                name='ITL',
                line=dict(color=PURPLE, width=2),
                marker=dict(size=6),
            ),
        ]

    safe = re.sub(r'[^a-zA-Z0-9_]', '_', run.dir_name)
    return ChartBuilder.line(
        traces,
        x_title='Percentile',
        y_title='Time (s)',
        div_id=f'chart-percentile-{safe}',
    )
