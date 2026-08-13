import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from evalscope.metrics.semantics import format_perf_value
from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.perf_models import BenchmarkSummary, PercentileResult
from evalscope.perf.utils.trace_metrics import TraceLevelSummary
from evalscope.perf.utils.workload_timeline import WorkloadThroughput
from evalscope.utils.logger import get_logger
from .perf_constants import Metrics, PercentileMetrics

logger = get_logger()


def _cell(field_key: str, value: Optional[float], include_unit: bool = False) -> str:
    """Format one perf table cell through the semantics registry."""
    return format_perf_value(value, field_key, include_unit=include_unit)


# ---------------------------------------------------------------------------
# Layer 0: Output encapsulation
# ---------------------------------------------------------------------------


class DualConsole:
    """Encapsulates writing to both a live console and a file console.

    Style kwargs are stripped for file output to avoid ANSI escape codes.
    """

    def __init__(self, console: Console, file_console: Console):
        self._console = console
        self._file_console = file_console

    def print(self, content, **kwargs):
        self._console.print(content, **kwargs)
        file_kwargs = {k: v for k, v in kwargs.items() if k != 'style'}
        self._file_console.print(content, **file_kwargs)


# ---------------------------------------------------------------------------
# Layer 1: Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Structured output of a result analysis pass.

    Attributes:
        rows:         Formatted summary rows (used by the Embedding detail table).
        total_tokens: Aggregate token count across all concurrency levels.
        total_time:   Aggregate test duration (seconds).
        all_results:  Original raw results retained for extra rendering.
        n_entries:    Number of processed concurrency/rate configs.
    """
    rows: List[Dict[str, str]]
    total_tokens: float
    total_time: float
    all_results: Any = None
    n_entries: int = 0


class ColSpec(NamedTuple):
    """Unified column specification tying a row-dict key to a Rich table header."""
    key: str
    header: str
    style: str = ''
    justify: str = 'right'


class EmbCol:
    """Column specs for the Embedding / Rerank Detailed Performance Metrics table."""
    CONCURRENCY = ColSpec('concurrency', 'Conc.', style='cyan')
    RATE = ColSpec('rate', 'Rate')
    RPS = ColSpec('rps', 'RPS')
    AVG_LATENCY = ColSpec('avg_latency', 'Avg\nLat.(s)')
    P99_LATENCY = ColSpec('p99_latency', 'P99\nLat.(s)')
    AVG_INPUT_TPS = ColSpec('avg_input_tps', 'Avg\nInp.TPS')
    P99_INPUT_TPS = ColSpec('p99_input_tps', 'P99\nInp.TPS')
    AVG_INPUT_TOKENS = ColSpec('avg_input_tokens', 'Avg\nInp.Tok')
    SUCCESS_RATE = ColSpec('success_rate', 'Success\nRate', style='green')

    ALL: List[ColSpec] = [
        CONCURRENCY,
        RATE,
        RPS,
        AVG_LATENCY,
        P99_LATENCY,
        AVG_INPUT_TPS,
        P99_INPUT_TPS,
        AVG_INPUT_TOKENS,
        SUCCESS_RATE,
    ]


# ---------------------------------------------------------------------------
# Layer 2: Analyzers
# ---------------------------------------------------------------------------


class BaseResultAnalyzer:
    """Static utilities for parsing raw benchmark results.

    Subclasses implement ``analyze()`` with API-specific logic.
    """

    @staticmethod
    def _parse_entry(entry: dict) -> Optional[Tuple[BenchmarkSummary, PercentileResult]]:
        """Parse a single result entry dict into typed objects. Returns ``None`` on failure."""
        raw_metrics = entry.get('metrics', {})
        raw_perc = entry.get('percentiles', {})
        summary = (
            raw_metrics if isinstance(raw_metrics, BenchmarkSummary) else BenchmarkSummary.from_dict(raw_metrics)
        )
        percentiles = (
            raw_perc if isinstance(raw_perc, PercentileResult) else
            PercentileResult.from_transposed(raw_perc) if isinstance(raw_perc, dict) and raw_perc else
            PercentileResult.from_list(raw_perc) if isinstance(raw_perc, list) else PercentileResult()
        )
        if summary is None:
            return None
        return summary, percentiles

    @staticmethod
    def normalize(all_results) -> list:
        """Normalize input to a list of ``(BenchmarkSummary, PercentileResult)`` tuples."""
        if isinstance(all_results, dict):
            pairs = []
            for v in all_results.values():
                if not isinstance(v, dict):
                    continue
                parsed = BaseResultAnalyzer._parse_entry(v)
                if parsed is not None:
                    pairs.append(parsed)
            return pairs
        return all_results

    @staticmethod
    def _get_p99(percentiles: PercentileResult, metric_field: str) -> Optional[float]:
        val = percentiles.get_p99(metric_field)
        return val if val != 0.0 else None

    def analyze(self, all_results) -> AnalysisResult:
        raise NotImplementedError


class LLMResultAnalyzer(BaseResultAnalyzer):
    """Analyzer for LLM text-generation APIs."""

    def analyze(self, all_results) -> AnalysisResult:
        total_tokens = 0.0
        total_time = 0.0
        n = 0
        for summary, _ in self.normalize(all_results):
            total_tokens += summary.avg_output_tokens * summary.succeed_requests
            total_time += summary.time_taken
            n += 1
        return AnalysisResult(
            rows=[],
            total_tokens=total_tokens,
            total_time=total_time,
            all_results=all_results,
            n_entries=n,
        )


class EmbeddingResultAnalyzer(BaseResultAnalyzer):
    """Analyzer for Embedding / Rerank APIs."""

    @staticmethod
    def _sort_rows(rows: List[Dict[str, str]]) -> None:
        """Sort rows ascending by (concurrency, rate) using EmbCol keys."""
        rows.sort(
            key=lambda x: (
                float(x[EmbCol.CONCURRENCY.key]) if x[EmbCol.CONCURRENCY.key] != 'INF' else float('inf'),
                float(x[EmbCol.RATE.key]) if x[EmbCol.RATE.key] != 'INF' else float('inf'),
            )
        )

    def analyze(self, all_results) -> AnalysisResult:
        rows: List[Dict[str, str]] = []
        total_tokens = 0.0
        total_time = 0.0

        for summary, percentiles in self.normalize(all_results):
            try:
                row = self._build_row(summary, percentiles)
                rows.append(row)
                total_tokens += summary.avg_input_tokens * summary.succeed_requests
                total_time += summary.time_taken
            except Exception as e:
                logger.warning(f'Warning: Error processing result entry: {e}')

        if rows:
            self._sort_rows(rows)

        return AnalysisResult(
            rows=rows,
            total_tokens=total_tokens,
            total_time=total_time,
            all_results=all_results,
            n_entries=len(rows),
        )

    def _build_row(self, summary: BenchmarkSummary, percentiles: PercentileResult) -> Dict[str, str]:
        concurrency = summary.concurrency
        rate = summary.request_rate
        rps = summary.request_throughput
        avg_latency = summary.avg_latency
        p99_latency = self._get_p99(percentiles, 'latency')
        success_rate = summary.success_rate

        avg_input_tps = summary.input_token_throughput
        p99_input_tps = self._get_p99(percentiles, 'input_throughput') or 0.0
        avg_input_tokens = summary.avg_input_tokens

        if any(x is None for x in [concurrency, rps, avg_latency, p99_latency]):
            raise ValueError(f'Test results for concurrency {concurrency} contain invalid data, skipped')

        return {
            EmbCol.CONCURRENCY.key: 'INF' if concurrency == -1 else str(int(concurrency)),
            EmbCol.RATE.key: _cell(Metrics.REQUEST_RATE, rate) if rate != -1 else 'INF',
            EmbCol.RPS.key: _cell(Metrics.REQUEST_THROUGHPUT, rps) if rps is not None else 'N/A',
            EmbCol.AVG_LATENCY.key: _cell(Metrics.AVERAGE_LATENCY, avg_latency) if avg_latency is not None else 'N/A',
            EmbCol.P99_LATENCY.key: _cell(PercentileMetrics.LATENCY, p99_latency) if p99_latency is not None else 'N/A',
            EmbCol.AVG_INPUT_TPS.key: _cell(Metrics.INPUT_TOKEN_THROUGHPUT, avg_input_tps)
            if avg_input_tps is not None else 'N/A',
            EmbCol.P99_INPUT_TPS.key: _cell(PercentileMetrics.INPUT_THROUGHPUT, p99_input_tps),
            EmbCol.AVG_INPUT_TOKENS.key: _cell(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, avg_input_tokens)
            if avg_input_tokens is not None else 'N/A',
            EmbCol.SUCCESS_RATE.key: _cell('success_rate', success_rate, include_unit=True)
            if success_rate is not None else 'N/A',
        }


# ---------------------------------------------------------------------------
# Layer 3: Renderers  (Template Method pattern)
# ---------------------------------------------------------------------------


class BaseSummaryRenderer(ABC):
    """Renders a performance summary report to a :class:`DualConsole`.

    Subclasses must implement ``_render_title``, ``_get_token_stat_rows``, and ``render``.
    The shared ``_render_basic_info`` is available for reuse.
    """

    @abstractmethod
    def _render_title(self, dc: DualConsole):
        """Render the top-level report title panel."""

    @abstractmethod
    def _get_token_stat_rows(self, result: AnalysisResult, args: Arguments) -> List[tuple]:
        """Return the two token-stat rows for the basic-info table."""

    @abstractmethod
    def render(self, result: AnalysisResult, args: Arguments, dc: DualConsole):
        """Render the complete report."""

    def _render_basic_info(self, result: AnalysisResult, args: Arguments, dc: DualConsole):
        basic_info = Table(show_header=False, width=80)
        basic_info.add_column('Name', style='cyan', width=25)
        basic_info.add_column('Value', style='green', width=55)

        basic_info.add_row('Model', args.model_id)
        basic_info.add_row('Test Dataset', args.dataset)
        basic_info.add_row('API Type', args.api)
        for label, value in self._get_token_stat_rows(result, args):
            basic_info.add_row(label, value)
        basic_info.add_row('Total Test Time', _cell(Metrics.TIME_TAKEN_FOR_TESTS, result.total_time, include_unit=True))
        basic_info.add_row('Output Path', args.outputs_dir)

        dc.print('\nBasic Information:')
        dc.print(basic_info)


class LLMSummaryRenderer(BaseSummaryRenderer):
    """Renderer for LLM text-generation performance reports.

    Outputs four tables:
    1. Performance Overview — one row per config (scalar metrics)
    2. Per-Request Metrics — vertical metric rows with percentile columns
    3. Per-Trace Metrics — multi-turn trace-level distributions (if available)
    4. Workload Throughput — time-window token throughput (if available)
    """

    def _render_title(self, dc: DualConsole):
        dc.print(Panel(Text('Performance Test Summary Report', style='bold'), width=80))

    def _get_token_stat_rows(self, result: AnalysisResult, args: Arguments) -> List[tuple]:
        total_tokens = result.total_tokens
        total_time = result.total_time
        return [
            ('Total Generated', f'{total_tokens:,} tokens'),
            (
                'Avg Output Rate', _cell(Metrics.OUTPUT_TOKEN_THROUGHPUT, total_tokens
                                         / total_time, include_unit=True) if total_time > 0 else 'N/A'
            ),
        ]

    def render(self, result: AnalysisResult, args: Arguments, dc: DualConsole):
        self._render_title(dc)
        self._render_basic_info(result, args, dc)
        if result.all_results is not None:
            entries = self._collect_entries(result.all_results)
            self._render_overview_table(entries, dc)
            self._render_per_request_metrics(entries, dc)
            self._render_per_trace_metrics(entries, dc)
            self._render_workload_throughput(entries, dc)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_entries(
        all_results,
    ) -> List[Tuple[BenchmarkSummary, PercentileResult, Optional[TraceLevelSummary], Optional[WorkloadThroughput]]]:
        """Collect per-config data tuples sorted by (concurrency, rate)."""
        if not isinstance(all_results, dict):
            return []
        entries: List[Tuple[BenchmarkSummary, PercentileResult, Optional[TraceLevelSummary],
                            Optional[WorkloadThroughput]]] = []
        for entry in all_results.values():
            if not isinstance(entry, dict):
                continue
            parsed = BaseResultAnalyzer._parse_entry(entry)
            if parsed is None:
                continue
            summary, percentiles = parsed
            trace_summary = entry.get('trace_summary')
            if not isinstance(trace_summary, TraceLevelSummary):
                trace_summary = None
            workload = entry.get('workload_throughput')
            if not isinstance(workload, WorkloadThroughput):
                workload = None
            entries.append((summary, percentiles, trace_summary, workload))
        entries.sort(
            key=lambda x: (
                x[0].concurrency if x[0].concurrency != -1 else float('inf'),
                x[0].request_rate if x[0].request_rate != -1 else float('inf'),
            )
        )
        return entries

    @staticmethod
    def _conc_rate(summary: BenchmarkSummary) -> Tuple[str, str]:
        """Format concurrency and rate display values."""
        conc = '-' if summary.concurrency == -1 else str(int(summary.concurrency))
        rate = '-' if summary.request_rate == -1 else _cell(Metrics.REQUEST_RATE, summary.request_rate)
        return conc, rate

    # ------------------------------------------------------------------
    # Table 1: Performance Overview
    # ------------------------------------------------------------------

    def _render_overview_table(self, entries: list, dc: DualConsole) -> None:
        if not entries:
            return

        has_traces = any(ts is not None and not ts.is_empty() for _, _, ts, _ in entries)

        table = Table(
            title='Performance Overview',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )
        table.add_column('Conc.', justify='right', style='cyan')
        table.add_column('Rate', justify='right')
        table.add_column('Num', justify='right')
        table.add_column('RPS', justify='right')
        table.add_column('Gen/s', justify='right')
        table.add_column('Success', justify='right', style='green')
        if has_traces:
            table.add_column('Traces', justify='right')

        for summary, _, trace_summary, _ in entries:
            conc, rate = self._conc_rate(summary)
            row = [
                conc,
                rate,
                _cell(Metrics.TOTAL_REQUESTS, summary.total_requests),
                _cell(Metrics.REQUEST_THROUGHPUT, summary.request_throughput),
                _cell(Metrics.OUTPUT_TOKEN_THROUGHPUT, summary.output_token_throughput),
                _cell('success_rate', summary.success_rate, include_unit=True),
            ]
            if has_traces:
                n = trace_summary.n_traces if trace_summary and not trace_summary.is_empty() else '-'
                row.append(str(n))
            sr = summary.success_rate
            style = 'green' if sr >= 95 else 'yellow' if sr >= 80 else 'red'
            table.add_row(*row, style=style)

        dc.print('\n')
        dc.print(table)

    # ------------------------------------------------------------------
    # Table 2: Per-Request Metrics
    # ------------------------------------------------------------------

    def _render_per_request_metrics(self, entries: list, dc: DualConsole) -> None:
        if not entries:
            return

        has_ttft = any(s.avg_ttft for s, _, _, _ in entries)
        has_tpot = any(s.avg_tpot for s, _, _, _ in entries)
        has_output = any(s.avg_output_tokens for s, _, _, _ in entries)
        has_turns = any(s.avg_turns is not None and s.avg_turns > 0 for s, _, _, _ in entries)
        has_cache = any(s.avg_cached_percent is not None and s.avg_cached_percent >= 0 for s, _, _, _ in entries)
        has_first_ttft = any(s.avg_first_turn_ttft is not None for s, _, _, _ in entries)
        has_subseq_ttft = any(s.avg_subsequent_turn_ttft is not None for s, _, _, _ in entries)
        has_decode = has_tpot
        has_spec = any(
            s.avg_decoded_tokens_per_iter is not None and s.avg_decoded_tokens_per_iter > 0 for s, _, _, _ in entries
        )

        table = Table(
            title='Per-Request Metrics',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )
        table.add_column('Conc.', justify='right', style='cyan')
        table.add_column('Rate', justify='right')
        table.add_column('Metric', justify='left', style='cyan')
        table.add_column('avg', justify='right')
        table.add_column('p50', justify='right')
        table.add_column('p99', justify='right')
        table.add_column('max', justify='right')

        for i, (summary, percentiles, _, _) in enumerate(entries):
            if i > 0:
                table.add_section()

            conc, rate = self._conc_rate(summary)
            first_row = True

            def _add(
                metric: str,
                avg_val: Optional[float],
                perc_field: Optional[str] = None,
                field_key: Optional[str] = None,
            ) -> None:
                nonlocal first_row
                c, r = (conc, rate) if first_row else ('', '')
                first_row = False
                semantics_key = field_key or metric
                avg_str = _cell(semantics_key, avg_val) if avg_val is not None else '-'
                p50_str = p99_str = max_str = '-'
                if perc_field:
                    p50 = percentiles.get_p('50%', perc_field)
                    p99 = percentiles.get_p('99%', perc_field)
                    pmax = percentiles.get_p('max', perc_field)
                    if p50 != 0.0:
                        p50_str = _cell(semantics_key, p50)
                    if p99 != 0.0:
                        p99_str = _cell(semantics_key, p99)
                    if pmax != 0.0:
                        max_str = _cell(semantics_key, pmax)
                table.add_row(c, r, metric, avg_str, p50_str, p99_str, max_str)

            def _add_scalar(
                metric: str,
                value: Optional[float],
                field_key: str,
                include_unit: bool = False,
            ) -> None:
                nonlocal first_row
                c, r = (conc, rate) if first_row else ('', '')
                first_row = False
                val_str = _cell(field_key, value, include_unit=include_unit) if value is not None else '-'
                table.add_row(c, r, metric, val_str, '-', '-', '-')

            _add(PercentileMetrics.LATENCY, summary.avg_latency, 'latency')
            if has_ttft:
                _add(PercentileMetrics.TTFT, summary.avg_ttft, 'ttft')
            if has_tpot:
                _add(PercentileMetrics.TPOT, summary.avg_tpot, 'tpot')
            _add('Input Tokens', summary.avg_input_tokens, 'input_tokens', PercentileMetrics.INPUT_TOKENS)
            if has_output:
                _add('Output Tokens', summary.avg_output_tokens, 'output_tokens', PercentileMetrics.OUTPUT_TOKENS)

            if has_turns:
                _add_scalar('Turns/Req', summary.avg_turns, Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST)
            if has_cache:
                v = summary.avg_cached_percent
                _add_scalar(
                    'Cache Hit (%)',
                    v if v is not None and v >= 0 else None,
                    Metrics.AVERAGE_CACHED_PERCENT,
                    include_unit=True,
                )
            if has_first_ttft:
                _add_scalar('1st-Turn TTFT (ms)', summary.avg_first_turn_ttft, Metrics.AVERAGE_FIRST_TURN_TTFT)
            if has_subseq_ttft:
                _add_scalar('Subseq. TTFT (ms)', summary.avg_subsequent_turn_ttft, Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT)
            if has_decode:
                tpot = summary.avg_tpot
                tps = (1000.0 / tpot) if tpot and tpot > 0 else None
                _add_scalar('Decode toks/s', tps, Metrics.OUTPUT_TOKEN_THROUGHPUT)
            if has_spec:
                _add_scalar(
                    'Decoded Tok/Iter', summary.avg_decoded_tokens_per_iter, Metrics.AVERAGE_DECODED_TOKENS_PER_ITER
                )
                _add_scalar(
                    'Spec. Accept Rate',
                    summary.approx_spec_acceptance_rate,
                    Metrics.APPROX_SPECULATIVE_ACCEPTANCE_RATE,
                    include_unit=True
                )

        dc.print('\n')
        dc.print(table)

    # ------------------------------------------------------------------
    # Table 3: Per-Trace Metrics
    # ------------------------------------------------------------------

    def _render_per_trace_metrics(self, entries: list, dc: DualConsole) -> None:
        trace_entries = [(summary, ts) for summary, _, ts, _ in entries if ts is not None and not ts.is_empty()]
        if not trace_entries:
            return

        table = Table(
            title='Per-Trace Metrics',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )
        table.add_column('Conc.', justify='right', style='cyan')
        table.add_column('Rate', justify='right')
        table.add_column('Metric', justify='left', style='cyan')
        for stat in ('mean', 'p50', 'p90', 'p99', 'max'):
            table.add_column(stat, justify='right')

        for i, (summary, trace_summary) in enumerate(trace_entries):
            if i > 0:
                table.add_section()
            conc, rate = self._conc_rate(summary)
            first_row = True
            for row in trace_summary.rows:
                c, r = (conc, rate) if first_row else ('', '')
                first_row = False
                table.add_row(
                    c,
                    r,
                    row.metric,
                    _cell(row.metric, row.mean),
                    _cell(row.metric, row.p50),
                    _cell(row.metric, row.p90),
                    _cell(row.metric, row.p99),
                    _cell(row.metric, row.max),
                )

        dc.print('\n')
        dc.print(table)

    # ------------------------------------------------------------------
    # Table 4: Workload Throughput
    # ------------------------------------------------------------------

    def _render_workload_throughput(self, entries: list, dc: DualConsole) -> None:
        wl_entries = [(summary, wl) for summary, _, _, wl in entries if wl is not None and not wl.is_empty()]
        if not wl_entries:
            return

        # Assumes last_window_s and warmup_frac are identical across all sweep configs
        first_wl = wl_entries[0][1]
        last_label = f'Last {int(first_wl.last_window_s)}s'
        steady_label = f'Steady (drop {int(first_wl.warmup_frac * 100)}%)'

        table = Table(
            title='Workload Throughput',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )
        table.add_column('Conc.', justify='right', style='cyan')
        table.add_column('Rate', justify='right')
        table.add_column('Metric (tok/s)', justify='left', style='cyan')
        table.add_column('Overall', justify='right')
        table.add_column(last_label, justify='right')
        table.add_column(steady_label, justify='right', style='green')

        for i, (summary, workload) in enumerate(wl_entries):
            if i > 0:
                table.add_section()
            conc, rate = self._conc_rate(summary)
            first_row = True
            for row in workload.rows:
                c, r = (conc, rate) if first_row else ('', '')
                first_row = False
                table.add_row(
                    c,
                    r,
                    row.metric,
                    _cell(Metrics.OUTPUT_TOKEN_THROUGHPUT, row.overall),
                    _cell(Metrics.OUTPUT_TOKEN_THROUGHPUT, row.last_window),
                    _cell(Metrics.OUTPUT_TOKEN_THROUGHPUT, row.steady_state),
                )

        dc.print('\n')
        dc.print(table)


class EmbeddingSummaryRenderer(BaseSummaryRenderer):
    """Renderer for Embedding / Rerank performance reports."""

    def _render_title(self, dc: DualConsole):
        dc.print(Panel(Text('Embedding/Rerank Performance Test Summary', style='bold'), width=80))

    def _get_token_stat_rows(self, result: AnalysisResult, args: Arguments) -> List[tuple]:
        total_tokens = result.total_tokens
        total_time = result.total_time
        return [
            ('Total Input Tokens', f'{total_tokens:,.0f} tokens'),
            (
                'Avg Input Rate', _cell(Metrics.INPUT_TOKEN_THROUGHPUT, total_tokens
                                        / total_time, include_unit=True) if total_time > 0 else 'N/A'
            ),
        ]

    def render(self, result: AnalysisResult, args: Arguments, dc: DualConsole):
        self._render_title(dc)
        self._render_basic_info(result, args, dc)
        self._render_detail_table(result, dc)

    def _render_detail_table(self, result: AnalysisResult, dc: DualConsole):
        table = Table(
            title='Detailed Performance Metrics',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )
        for col in EmbCol.ALL:
            table.add_column(col.header, justify=col.justify, style=col.style)
        for row in result.rows:
            try:
                sr = float(row[EmbCol.SUCCESS_RATE.key].rstrip('%'))
                style = 'green' if sr >= 95 else 'yellow' if sr >= 80 else 'red'
                table.add_row(*row.values(), style=style)
            except ValueError as e:
                dc.print(f'Warning: Error processing row: {e}', style='bold red')
        dc.print('\n')
        dc.print(table)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_summary(all_results, args: Arguments):
    """Print the test-results summary to stdout and save it to a text file."""
    is_emb = Metrics.is_embedding_or_rerank(args.api)
    analyzer = EmbeddingResultAnalyzer() if is_emb else LLMResultAnalyzer()
    renderer = EmbeddingSummaryRenderer() if is_emb else LLMSummaryRenderer()

    result = analyzer.analyze(all_results)
    if result.n_entries == 0:
        logger.warning('No available test result data to display')
        return

    console = Console(width=120)
    summary_file = os.path.join(args.outputs_dir, 'performance_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        file_console = Console(file=f, width=120, force_terminal=False)
        dc = DualConsole(console, file_console)
        renderer.render(result, args, dc)

    logger.info(f'Performance summary saved to: {summary_file}')
