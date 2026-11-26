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

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from evalscope.perf.utils.perf_models import RunData

# ── Design tokens (must match report.css :root variables) ─────────────────────
ACCENT = '#63b3ed'
PURPLE = '#a78bfa'
GREEN = '#34d399'
YELLOW = '#fbbf24'
RED = '#f87171'

# Hide the modebar and Plotly logo so embedded charts read as clean cards.
PLOTLY_CONFIG: Dict[str, Any] = {'responsive': True, 'displayModeBar': False, 'displaylogo': False}

_GRID = dict(
    gridcolor='rgba(99,179,237,0.08)',
    zerolinecolor='rgba(99,179,237,0.12)',
)

_LAYOUT_BASE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    hovermode='x unified',
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


def _is_fully_transparent(color: Any) -> bool:
    """Return True when *color* renders fully transparent (alpha 0) in plotly.

    Covers the ``rgba(r,g,b,0)`` family (spaces and float alpha allowed), the
    ``transparent`` keyword, and 8-digit hex with a zero alpha channel.
    """
    if not isinstance(color, str):
        return False
    text = color.strip().lower()
    if text == 'transparent':
        return True
    if text.startswith('rgba') and text.endswith(')'):
        parts = [p.strip() for p in text[text.find('(') + 1 : text.rfind(')')].split(',')]
        if len(parts) == 4:
            try:
                return float(parts[3]) == 0.0
            except ValueError:
                return False
    if text.startswith('#') and len(text) == 9:
        try:
            return int(text[7:9], 16) == 0
        except ValueError:
            return False
    return False


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
    def layout(theme: str = 'dark', **overrides: Any) -> dict:
        """Return a layout dict by merging *_LAYOUT_BASE* with *overrides*.

        *theme* selects the Plotly template ('plotly_white' for light, else
        'plotly_dark'); backgrounds stay transparent so charts blend into the
        host surface on both themes. An explicit ``template`` in *overrides*
        takes precedence.
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        merged = dict(_LAYOUT_BASE)
        # An explicit `template` override wins over the theme pick, and the
        # hoverlabel colors below must follow the *effective* template so a
        # dark template never gets a light tooltip box (or vice versa).
        template = overrides.get('template') or ('plotly_white' if theme == 'light' else 'plotly_dark')
        merged['template'] = template
        # Opaque unified-hover tooltip box. plotly.js defaults an unset
        # hoverlabel.bgcolor to paper_bgcolor for legend-less charts, which is
        # transparent here -> see-through tooltip. Resolve the colors
        # dynamically from the effective chart background (explicit override
        # > template > per-theme fallback) so future background changes are
        # picked up automatically. Border follows the template's legend border
        # (the plotly.js default for unified hover on charts with a legend),
        # falling back to plotly.js's own default border color '#444'.
        tpl_layout = (
            pio.templates[template].layout if isinstance(template, str) else go.layout.Template(template).layout
        )
        paper = overrides.get('paper_bgcolor') or getattr(tpl_layout, 'paper_bgcolor', None)
        if not paper or _is_fully_transparent(paper):  # unset or fully transparent
            paper = '#ffffff' if theme == 'light' else 'rgb(17,17,17)'
        border = getattr(getattr(tpl_layout, 'legend', None), 'bordercolor', None) or '#444444'
        merged['hoverlabel'] = dict(bgcolor=paper, bordercolor=border)
        if 'hoverlabel' in overrides:
            # Merge a partial caller override into the computed hoverlabel
            # instead of letting it replace the dict wholesale (which would
            # drop bgcolor/bordercolor and reintroduce the see-through
            # tooltip). Explicit caller keys still win per-key.
            merged['hoverlabel'].update(overrides['hoverlabel'])
            overrides = {key: value for key, value in overrides.items() if key != 'hoverlabel'}
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
        theme: str = 'dark',
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

        fig.update_layout(**cls.layout(theme=theme, **layout_kwargs))
        return cls.to_div(fig, div_id)

    @classmethod
    def scatter(
        cls,
        traces: list,
        x_title: str,
        y_title: str,
        div_id: str = '',
        theme: str = 'dark',
    ) -> str:
        """Build a ``Scatter`` (marker-only) chart and return an HTML div."""
        import plotly.graph_objects as go

        fig = go.Figure()
        for trace in traces:
            fig.add_trace(go.Scatter(**trace))

        fig.update_layout(
            **cls.layout(
                theme=theme,
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
        theme: str = 'dark',
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
                theme=theme,
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


def _x_axis(runs: 'List[RunData]'):
    """Return *(xs, x_title)* appropriate for the run mode.

    * open-loop  (``rate_*_number_*`` dirs) → xs = request_rate floats,
      x_title = ``'Rate (req/s)'``
    * closed-loop (``parallel_*_number_*`` dirs) → xs = concurrency integers,
      x_title = ``'Concurrency'``
    """
    is_open = any(r.is_open_loop for r in runs)
    if is_open:
        return [r.summary.request_rate for r in runs], 'Rate (req/s)'
    return [r.parallel for r in runs], 'Concurrency'


# ---------------------------------------------------------------------------
# Per-sweep charts (one data point per run)
# ---------------------------------------------------------------------------


def build_latency_chart(runs: List[RunData], theme: str = 'dark') -> str:
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
    return ChartBuilder.line(traces, x_title=x_title, y_title='Latency (s)', div_id='chart-latency', theme=theme)


def build_ttft_chart(runs: List[RunData], theme: str = 'dark') -> str:
    """Line chart: Avg + P99 Time-To-First-Token vs concurrency / rate (LLM only)."""
    xs, x_title = _x_axis(runs)
    traces = [
        dict(
            x=xs,
            y=[r.summary.avg_ttft for r in runs],
            mode='lines+markers',
            name='Avg TTFT',
            line=dict(color=GREEN, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('ttft') for r in runs],
            mode='lines+markers',
            name='P99 TTFT',
            line=dict(color=YELLOW, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title=x_title, y_title='TTFT (ms)', div_id='chart-ttft', theme=theme)


def build_tpot_chart(runs: List[RunData], theme: str = 'dark') -> str:
    """Line chart: Avg + P99 Time-Per-Output-Token vs concurrency / rate (LLM only)."""
    xs, x_title = _x_axis(runs)
    traces = [
        dict(
            x=xs,
            y=[r.summary.avg_tpot for r in runs],
            mode='lines+markers',
            name='Avg TPOT',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
        ),
        dict(
            x=xs,
            y=[r.get_p99('tpot') for r in runs],
            mode='lines+markers',
            name='P99 TPOT',
            line=dict(color=RED, width=2, dash='dash'),
            marker=dict(size=8),
        ),
    ]
    return ChartBuilder.line(traces, x_title=x_title, y_title='TPOT (ms)', div_id='chart-tpot', theme=theme)


def build_rps_chart(runs: List[RunData], theme: str = 'dark') -> str:
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
    return ChartBuilder.line(traces, x_title=x_title, y_title='Requests/sec', div_id='chart-rps', theme=theme)


def build_throughput_chart(runs: List[RunData], is_embedding: bool, theme: str = 'dark') -> str:
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
        theme=theme,
    )


def build_success_chart(runs: List[RunData], theme: str = 'dark') -> str:
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
        extra_layout=dict(yaxis=dict(range=[0, 105], **_GRID)),
        theme=theme,
    )


# ---------------------------------------------------------------------------
# Cross-run comparison chart (one line per selected run directory)
# ---------------------------------------------------------------------------

_COMPARE_PALETTE = [ACCENT, GREEN, PURPLE, YELLOW, RED, '#f472b6', '#22d3ee', '#a3e635']

_COMPARE_Y_TITLE = {
    'latency': 'Latency (s)',
    'ttft': 'TTFT (ms)',
    'tpot': 'TPOT (ms)',
    'rps': 'Requests/sec',
    'throughput': 'Tokens/sec',
    'success': 'Success Rate (%)',
}


def _compare_y(metric: str, r: 'RunData', is_embedding: bool) -> float:
    """Return the y-value of *metric* for a single sweep sub-run."""
    s = r.summary
    if metric == 'latency':
        return s.avg_latency
    if metric == 'ttft':
        return s.avg_ttft
    if metric == 'tpot':
        return s.avg_tpot
    if metric == 'rps':
        return s.request_throughput
    if metric == 'success':
        return r.success_rate
    if metric == 'throughput':
        return s.input_token_throughput if is_embedding else s.output_token_throughput
    return 0.0


def _finite_metric(value: float) -> Optional[float]:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def build_compare_chart(
    series: 'List[Tuple[str, List[RunData]]]',
    metric: str,
    is_embedding: bool = False,
    theme: str = 'dark',
) -> str:
    """Compare sweeps on separate numeric axes for concurrency and request rate."""
    y_title = _COMPARE_Y_TITLE.get(metric, metric)
    modes = {run.is_open_loop for _, runs in series for run in runs}
    split_axes = len(modes) > 1
    traces = []
    for i, (display_name, runs) in enumerate(series):
        run_id = f'R{i + 1}'
        color = _COMPARE_PALETTE[i % len(_COMPARE_PALETTE)]
        by_mode: dict[bool, list[RunData]] = defaultdict(list)
        for run in runs:
            by_mode[run.is_open_loop].append(run)
        for index, (open_loop, sweep) in enumerate(by_mode.items()):
            xs, _ = _x_axis(sweep)
            traces.append(
                dict(
                    x=xs,
                    xaxis='x2' if split_axes and open_loop else 'x',
                    y=[_finite_metric(_compare_y(metric, run, is_embedding)) for run in sweep],
                    mode='markers' if len(xs) <= 1 else 'lines+markers',
                    name=f'{run_id} · {display_name}',
                    legendgroup=run_id,
                    showlegend=index == 0,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    connectgaps=False,
                    hovertemplate=f'%{{y}}<extra>{run_id}</extra>',
                )
            )

    # Share the metric (y) axis while keeping the two workload units on disjoint x domains.
    layout = (
        {
            'xaxis': dict(domain=[0, 0.46], dtick=1, **_GRID),
            'xaxis2': dict(domain=[0.54, 1], title='Rate (req/s)', anchor='y', **_GRID),
        }
        if split_axes
        else None
    )
    return ChartBuilder.line(
        traces,
        x_title='Rate (req/s)' if modes == {True} else 'Concurrency',
        y_title=y_title,
        div_id=f'chart-compare-{metric}',
        extra_layout=layout,
        theme=theme,
    )


def build_input_length_compare_chart(
    points: 'List[Tuple[int, int, RunData]]',
    metric: str,
    is_embedding: bool = False,
    theme: str = 'dark',
) -> str:
    """Plot configured input length with one cross-run trace per concurrency."""
    y_title = _COMPARE_Y_TITLE.get(metric, metric)
    lengths = sorted({length for length, _, _ in points})
    concurrencies = sorted({concurrency for _, concurrency, _ in points})
    traces = []
    for index, concurrency in enumerate(concurrencies):
        by_length = {length: run for length, point_concurrency, run in points if point_concurrency == concurrency}
        color = _COMPARE_PALETTE[index % len(_COMPARE_PALETTE)]
        traces.append(
            dict(
                x=lengths,
                y=[
                    _finite_metric(_compare_y(metric, by_length[length], is_embedding)) if length in by_length else None
                    for length in lengths
                ],
                mode='markers' if len(by_length) <= 1 else 'lines+markers',
                name=f'C={concurrency}',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                connectgaps=False,
                hovertemplate='%{y}<extra>%{fullData.name}</extra>',
            )
        )
    return ChartBuilder.line(
        traces,
        x_title='Configured input length',
        y_title=y_title,
        div_id=f'chart-compare-input-length-{metric}',
        extra_layout={'xaxis': dict(**_GRID)},
        theme=theme,
    )


# ---------------------------------------------------------------------------
# Per-run detail charts
# ---------------------------------------------------------------------------


def build_request_detail_tabs(run: 'RunData', is_embedding: bool, theme: str = 'dark') -> list:
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
    tabs.append(
        {
            'label': 'Latency',
            'chart': ChartBuilder.line(
                lat_traces,
                x_title='Request Index',
                y_title='Latency (s)',
                div_id=f'req-latency-{safe}',
                theme=theme,
            ),
        }
    )

    # ── Tab 2: TTFT / TPOT / ITL (milliseconds, LLM only) ───────────────────
    if not is_embedding:
        itl_y = [
            (sum(r.inter_token_latencies) / len(r.inter_token_latencies)) if r.inter_token_latencies else 0
            for r in sorted_reqs
        ]
        ttft_tpot_itl_traces = [
            dict(
                x=xs,
                y=[r.first_chunk_latency if r.first_chunk_latency is not None else 0 for r in sorted_reqs],
                mode='lines+markers',
                name='TTFT',
                line=dict(color=GREEN, width=1.5),
                marker=dict(size=4),
            ),
            dict(
                x=xs,
                y=[r.time_per_output_token if r.time_per_output_token is not None else 0 for r in sorted_reqs],
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
        tabs.append(
            {
                'label': 'TTFT / TPOT / ITL',
                'chart': ChartBuilder.line(
                    ttft_tpot_itl_traces,
                    x_title='Request Index',
                    y_title='Time (ms)',
                    div_id=f'req-ttft-tpot-itl-{safe}',
                    theme=theme,
                ),
            }
        )

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
    tabs.append(
        {
            'label': 'Tokens',
            'chart': ChartBuilder.line(
                tok_traces,
                x_title='Request Index',
                y_title='Token Count',
                div_id=f'req-tokens-{safe}',
                theme=theme,
            ),
        }
    )

    # ── Tab 4: Success ────────────────────────────────────────────────────
    tabs.append(
        {
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
                theme=theme,
            ),
        }
    )

    return tabs


def build_percentile_chart(run: 'RunData', is_embedding: bool, theme: str = 'dark') -> 'Tuple[str, str]':
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
        theme=theme,
    )

    # Chart 2: TTFT / TPOT / ITL (ms) — LLM only
    token_lat_chart = ''
    if not is_embedding:
        token_lat_traces = [
            dict(
                x=xs,
                y=[(row.ttft or 0) for row in run.percentiles.rows],
                mode='lines+markers',
                name='TTFT',
                line=dict(color=GREEN, width=2),
                marker=dict(size=6),
            ),
            dict(
                x=xs,
                y=[(row.tpot or 0) for row in run.percentiles.rows],
                mode='lines+markers',
                name='TPOT',
                line=dict(color=YELLOW, width=2),
                marker=dict(size=6),
            ),
            dict(
                x=xs,
                y=[(row.itl or 0) for row in run.percentiles.rows],
                mode='lines+markers',
                name='ITL',
                line=dict(color=PURPLE, width=2),
                marker=dict(size=6),
            ),
        ]
        token_lat_chart = ChartBuilder.line(
            token_lat_traces,
            x_title='Percentile',
            y_title='Time (ms)',
            div_id=f'chart-percentile-token-lat-{safe}',
            theme=theme,
        )

    return latency_chart, token_lat_chart
