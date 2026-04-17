import numpy as np
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import Any, List

from evalscope.perf.arguments import Arguments
from evalscope.utils.logger import get_logger
from .benchmark_util import Metrics
from .db_util import PercentileMetrics

logger = get_logger()

# ---------------------------------------------------------------------------
# Layer 0: Output encapsulation
# ---------------------------------------------------------------------------


class DualConsole:
    """Encapsulates writing to both a live console and a file console.

    Eliminates the need to pass two console objects through every function.
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
        rows:         Formatted summary rows.  Column layout is defined by the
                      concrete analyzer that produced them but the layout
                      contract is: col[0]=Concurrency, col[1]=Rate, col[2]=RPS,
                      col[3]=Avg Latency, col[-1]=Success Rate.
        total_tokens: Aggregate token count across all concurrency levels.
        total_time:   Aggregate test duration (seconds).
        all_results:  Original raw results retained for extra rendering
                      (e.g. the LLM Request Metrics table).
    """
    rows: List[List[str]]
    total_tokens: float
    total_time: float
    all_results: Any = None


# ---------------------------------------------------------------------------
# Layer 2: Analyzers
# ---------------------------------------------------------------------------


class BaseResultAnalyzer(ABC):
    """Extracts and formats metrics from raw benchmark results.

    Subclasses implement ``_process_one`` to handle API-specific fields.
    Common logic (normalization, p99 lookup, sorting) lives here.
    """

    @staticmethod
    def normalize(all_results) -> list:
        """Normalize input to a list of ``(metrics, percentile_metrics)`` tuples."""
        if isinstance(all_results, dict):
            return [(v['metrics'], v['percentiles'])
                    for v in all_results.values()
                    if 'metrics' in v and 'percentiles' in v]
        return all_results

    @staticmethod
    def _get_p99(percentile_metrics, key, percentiles):
        """Safely fetch the p99 value for *key* from *percentile_metrics*.

        Returns ``None`` when the key is absent or the index cannot be found.
        """
        data = percentile_metrics.get(key)
        if data is None:
            return None
        try:
            return data[percentiles.index('99%')]
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _sort_rows(rows: list):
        """Sort rows ascending by (concurrency, rate)."""
        rows.sort(key=lambda x: (float(x[0]), float(x[1]) if x[1] != 'INF' else float('inf')))

    def analyze(self, all_results) -> AnalysisResult:
        """Run analysis over all concurrency levels and return an :class:`AnalysisResult`."""
        rows: list = []
        total_tokens = 0.0
        total_time = 0.0

        for entry in self.normalize(all_results):
            try:
                row, tokens, elapsed = self._process_one(entry)
                rows.append(row)
                total_tokens += tokens
                total_time += elapsed
            except Exception as e:
                logger.warning(f'Warning: Error processing result entry: {e}')

        if rows:
            self._sort_rows(rows)

        return AnalysisResult(
            rows=rows,
            total_tokens=total_tokens,
            total_time=total_time,
            all_results=all_results,
        )

    @abstractmethod
    def _process_one(self, entry) -> tuple:
        """Process a single ``(metrics, percentile_metrics)`` entry.

        Returns:
            ``(formatted_row, token_count, elapsed_seconds)``

        Raises:
            Exception: on invalid or incomplete data (entry is skipped).
        """


class LLMResultAnalyzer(BaseResultAnalyzer):
    """Analyzer for LLM text-generation APIs."""

    def _process_one(self, entry) -> tuple:
        total_metrics, percentile_metrics = entry
        percentiles = percentile_metrics[PercentileMetrics.PERCENTILES]

        concurrency = total_metrics.get(Metrics.NUMBER_OF_CONCURRENCY, 0)
        rate = total_metrics.get(Metrics.REQUEST_RATE, 0)
        rps = total_metrics.get(Metrics.REQUEST_THROUGHPUT, 0)
        avg_latency = total_metrics.get(Metrics.AVERAGE_LATENCY, 0)
        p99_latency = self._get_p99(percentile_metrics, PercentileMetrics.LATENCY, percentiles)
        success_rate = (
            total_metrics.get(Metrics.SUCCEED_REQUESTS, 0) / total_metrics.get(Metrics.TOTAL_REQUESTS, 1)
        ) * 100

        avg_tps = total_metrics.get(Metrics.OUTPUT_TOKEN_THROUGHPUT, 0)
        avg_ttft = total_metrics.get(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, 0)
        p99_ttft = self._get_p99(percentile_metrics, PercentileMetrics.TTFT, percentiles)
        avg_tpot = total_metrics.get(Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, 0)
        p99_tpot = self._get_p99(percentile_metrics, PercentileMetrics.TPOT, percentiles)

        if any(x is None for x in [concurrency, rps, avg_latency, p99_latency]):
            raise ValueError(f'Test results for concurrency {concurrency} contain invalid data, skipped')

        row = [
            str(int(concurrency)),
            f'{rate:.2f}' if rate != -1 else 'INF',
            f'{rps:.2f}' if rps is not None else 'N/A',
            f'{avg_latency:.3f}' if avg_latency is not None else 'N/A',
            f'{p99_latency:.3f}' if p99_latency is not None else 'N/A',
            f'{avg_ttft:.3f}' if avg_ttft is not None else 'N/A',
            f'{p99_ttft:.3f}' if p99_ttft is not None else 'N/A',
            f'{avg_tpot:.3f}' if avg_tpot is not None else 'N/A',
            f'{p99_tpot:.3f}' if p99_tpot is not None else 'N/A',
            f'{avg_tps:.2f}' if avg_tps is not None else 'N/A',
            f'{success_rate:.1f}%' if success_rate is not None else 'N/A',
        ]
        tokens = (
            total_metrics.get(Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST, 0)
            * total_metrics.get(Metrics.SUCCEED_REQUESTS, 0)
        )
        elapsed = total_metrics.get(Metrics.TIME_TAKEN_FOR_TESTS, 0)
        return row, tokens, elapsed


class EmbeddingResultAnalyzer(BaseResultAnalyzer):
    """Analyzer for Embedding / Rerank APIs."""

    def _process_one(self, entry) -> tuple:
        total_metrics, percentile_metrics = entry
        percentiles = percentile_metrics[PercentileMetrics.PERCENTILES]

        concurrency = total_metrics.get(Metrics.NUMBER_OF_CONCURRENCY, 0)
        rate = total_metrics.get(Metrics.REQUEST_RATE, 0)
        rps = total_metrics.get(Metrics.REQUEST_THROUGHPUT, 0)
        avg_latency = total_metrics.get(Metrics.AVERAGE_LATENCY, 0)
        p99_latency = self._get_p99(percentile_metrics, PercentileMetrics.LATENCY, percentiles)
        success_rate = (
            total_metrics.get(Metrics.SUCCEED_REQUESTS, 0) / total_metrics.get(Metrics.TOTAL_REQUESTS, 1)
        ) * 100

        avg_input_tps = total_metrics.get(Metrics.INPUT_TOKEN_THROUGHPUT, 0)
        # Default to 0 when INPUT_THROUGHPUT percentile data is absent (mirrors original behaviour)
        p99_input_tps = self._get_p99(percentile_metrics, PercentileMetrics.INPUT_THROUGHPUT, percentiles) or 0.0
        avg_input_tokens = total_metrics.get(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, 0)

        if any(x is None for x in [concurrency, rps, avg_latency, p99_latency]):
            raise ValueError(f'Test results for concurrency {concurrency} contain invalid data, skipped')

        row = [
            str(int(concurrency)),
            f'{rate:.2f}' if rate != -1 else 'INF',
            f'{rps:.2f}' if rps is not None else 'N/A',
            f'{avg_latency:.3f}' if avg_latency is not None else 'N/A',
            f'{p99_latency:.3f}' if p99_latency is not None else 'N/A',
            f'{avg_input_tps:.2f}' if avg_input_tps is not None else 'N/A',
            f'{p99_input_tps:.2f}',
            f'{avg_input_tokens:.1f}' if avg_input_tokens is not None else 'N/A',
            f'{success_rate:.1f}%' if success_rate is not None else 'N/A',
        ]
        tokens = (
            total_metrics.get(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, 0)
            * total_metrics.get(Metrics.SUCCEED_REQUESTS, 0)
        )
        elapsed = total_metrics.get(Metrics.TIME_TAKEN_FOR_TESTS, 0)
        return row, tokens, elapsed


# ---------------------------------------------------------------------------
# Layer 3: Renderers  (Template Method pattern)
# ---------------------------------------------------------------------------


def _make_detail_table() -> Table:
    """Create a Rich Table with standard styling for detailed performance metrics."""
    return Table(
        title='Detailed Performance Metrics',
        show_header=True,
        header_style='bold cyan',
        border_style='blue',
        pad_edge=False,
        expand=False,
    )


class BaseSummaryRenderer(ABC):
    """Renders a performance summary report to a :class:`DualConsole`.

    Uses the **Template Method** pattern.  The fixed rendering order is:

    1. Title panel
    2. Basic information table
    3. Detailed per-concurrency metrics table
    4. Extra tables  *(hook — default is no-op)*
    5. Best-configuration summary + recommendations
    """

    def render(self, result: AnalysisResult, args: Arguments, dc: DualConsole):
        self._render_title(dc)
        self._render_basic_info(result, args, dc)
        self._render_detail_table(result, dc)
        self._render_extra_tables(result, dc)
        self._render_recommendations(result, dc)

    # ------------------------------------------------------------------
    # Abstract steps

    @abstractmethod
    def _render_title(self, dc: DualConsole):
        """Render the top-level report title panel."""

    @abstractmethod
    def _get_token_stat_rows(self, result: AnalysisResult, args: Arguments) -> List[tuple]:
        """Return the two token-stat rows that differ between LLM and Embedding.

        Each element is a ``(label, value)`` tuple inserted into the basic-info
        table, e.g.:
            ``[('Total Generated', '1,234 tokens'), ('Avg Output Rate', '56.78 tokens/sec')]``
        """

    @abstractmethod
    def _render_detail_table(self, result: AnalysisResult, dc: DualConsole):
        """Render the per-concurrency performance metrics table."""

    # ------------------------------------------------------------------
    # Concrete / shared steps

    def _render_basic_info(self, result: AnalysisResult, args: Arguments, dc: DualConsole):
        basic_info = Table(show_header=False, width=80)
        basic_info.add_column('Name', style='cyan', width=25)
        basic_info.add_column('Value', style='green', width=55)

        basic_info.add_row('Model', args.model_id)
        basic_info.add_row('Test Dataset', args.dataset)
        basic_info.add_row('API Type', args.api)
        for label, value in self._get_token_stat_rows(result, args):
            basic_info.add_row(label, value)
        basic_info.add_row('Total Test Time', f'{result.total_time:.2f} seconds')
        basic_info.add_row('Output Path', args.outputs_dir)

        dc.print('\nBasic Information:')
        dc.print(basic_info)

    def _render_extra_tables(self, result: AnalysisResult, dc: DualConsole):
        """Hook for additional tables.  Default implementation is a no-op."""

    def _render_recommendations(self, result: AnalysisResult, dc: DualConsole):
        """Render best-configuration summary and textual recommendations."""
        rows = result.rows
        try:
            # col[2] = RPS, col[3] = Avg Latency — consistent for both API types
            best_rps_idx = np.argmax([float(row[2]) if row[2] != 'N/A' else -1 for row in rows])
            best_latency_idx = np.argmin([float(row[3]) if row[3] != 'N/A' else float('inf') for row in rows])

            perf_info = Table(title='Best Performance Configuration', show_header=False, box=None, width=60)
            perf_info.add_column('Metric', style='cyan', width=20)
            perf_info.add_column('Value', style='green', width=40)
            perf_info.add_row(
                'Highest RPS',
                f'Concurrency {rows[best_rps_idx][0]} ({rows[best_rps_idx][2]} req/sec)',
            )
            perf_info.add_row(
                'Lowest Latency',
                f'Concurrency {rows[best_latency_idx][0]} ({rows[best_latency_idx][3]} seconds)',
            )
            dc.print('\n')
            dc.print(perf_info)

            recommendations = []
            if best_rps_idx == len(rows) - 1:
                recommendations.append(
                    'The system seems not to have reached its performance bottleneck, try higher concurrency'
                )
            elif best_rps_idx == 0:
                recommendations.append('Consider lowering concurrency, current load may be too high')
            else:
                recommendations.append(f'Optimal concurrency range is around {rows[best_rps_idx][0]}')

            # col[-1] = Success Rate — consistent for both API types
            success_rate_str = rows[-1][-1].rstrip('%')
            success_rate = float(success_rate_str) if success_rate_str != 'N/A' else 0
            if success_rate < 95:
                recommendations.append(
                    'Success rate is low at high concurrency, check system resources or reduce concurrency'
                )

            dc.print('\nPerformance Recommendations:', style='bold cyan')
            for rec in recommendations:
                dc.print(f'• {rec}', style='yellow')

        except Exception as e:
            dc.print(f'Warning: Error generating performance analysis: {e}', style='bold red')

    @staticmethod
    def _fill_table_rows(table: Table, rows: List[List[str]], dc: DualConsole):
        """Add *rows* to *table* with success-rate-based row colouring."""
        for row in rows:
            try:
                success_rate = float(row[-1].rstrip('%'))
                row_style = 'green' if success_rate >= 95 else 'yellow' if success_rate >= 80 else 'red'
                table.add_row(*row, style=row_style)
            except ValueError as e:
                dc.print(f'Warning: Error processing row: {e}', style='bold red')


class LLMSummaryRenderer(BaseSummaryRenderer):
    """Renderer for LLM text-generation performance reports."""

    def _render_title(self, dc: DualConsole):
        dc.print(Panel(Text('Performance Test Summary Report', style='bold'), width=80))

    def _get_token_stat_rows(self, result: AnalysisResult, args: Arguments) -> List[tuple]:
        total_tokens = result.total_tokens
        total_time = result.total_time
        return [
            ('Total Generated', f'{total_tokens:,} tokens'),
            ('Avg Output Rate', f'{total_tokens / total_time:.2f} tokens/sec' if total_time > 0 else 'N/A'),
        ]

    def _render_detail_table(self, result: AnalysisResult, dc: DualConsole):
        table = _make_detail_table()
        table.add_column('Conc.', justify='right', style='cyan')
        table.add_column('Rate', justify='right')
        table.add_column('RPS', justify='right')
        table.add_column('Avg Lat.(s)', justify='right')
        table.add_column('P99 Lat.(s)', justify='right')
        table.add_column('Avg TTFT(s)', justify='right')
        table.add_column('P99 TTFT(s)', justify='right')
        table.add_column('Avg TPOT(s)', justify='right')
        table.add_column('P99 TPOT(s)', justify='right')
        table.add_column('Gen. toks/s', justify='right')
        table.add_column('Success Rate', justify='right', style='green')

        self._fill_table_rows(table, result.rows, dc)
        dc.print('\n')
        dc.print(table)

    def _render_extra_tables(self, result: AnalysisResult, dc: DualConsole):
        if result.all_results is not None:
            self._render_request_metrics(result.all_results, dc)

    def _render_request_metrics(self, all_results, dc: DualConsole):
        """Render the per-concurrency request-level token / turn metrics table."""
        results = BaseResultAnalyzer.normalize(all_results)
        if not results:
            return

        rows = []
        has_turns = False
        has_cache = False

        for entry in results:
            total_metrics, percentile_metrics = entry
            percentiles = percentile_metrics.get(PercentileMetrics.PERCENTILES, [])

            concurrency = total_metrics.get(Metrics.NUMBER_OF_CONCURRENCY, 0)
            num_reqs = total_metrics.get(Metrics.TOTAL_REQUESTS)
            avg_in = total_metrics.get(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST)
            avg_out = total_metrics.get(Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST)
            avg_turns = total_metrics.get(Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST)
            avg_cache = total_metrics.get(Metrics.AVERAGE_CACHED_PERCENT)

            p99_in, p99_out = None, None
            try:
                idx99 = percentiles.index('99%')
                in_data = percentile_metrics.get(PercentileMetrics.INPUT_TOKENS)
                if in_data:
                    p99_in = in_data[idx99]
                out_data = percentile_metrics.get(PercentileMetrics.OUTPUT_TOKENS)
                if out_data:
                    p99_out = out_data[idx99]
            except (ValueError, IndexError):
                pass

            if avg_turns is not None and avg_turns > 0:
                has_turns = True
            if avg_cache is not None and avg_cache > 0:
                has_cache = True

            rows.append({
                'conc': str(int(concurrency)),
                'num_reqs': str(int(num_reqs)) if num_reqs is not None else 'N/A',
                'avg_in': f'{avg_in:.1f}' if avg_in is not None else 'N/A',
                'p99_in': f'{p99_in:.1f}' if p99_in is not None else 'N/A',
                'avg_out': f'{avg_out:.1f}' if avg_out is not None else 'N/A',
                'p99_out': f'{p99_out:.1f}' if p99_out is not None else 'N/A',
                'avg_turns': f'{avg_turns:.2f}' if (avg_turns is not None and avg_turns > 0) else None,
                'avg_cache': f'{avg_cache:.1f}%' if (avg_cache is not None and avg_cache > 0) else None,
            })

        if not rows:
            return

        req_table = Table(
            title='Request Metrics',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )
        req_table.add_column('Conc.', justify='right', style='cyan')
        req_table.add_column('Num Reqs', justify='right')
        req_table.add_column('Avg In Toks', justify='right')
        req_table.add_column('P99 In Toks', justify='right')
        req_table.add_column('Avg Out Toks', justify='right')
        req_table.add_column('P99 Out Toks', justify='right')
        if has_turns:
            req_table.add_column('Avg Turns/Req', justify='right')
        if has_cache:
            req_table.add_column('Approx Cache Hit', justify='right', style='green')

        for r in rows:
            row_data = [r['conc'], r['num_reqs'], r['avg_in'], r['p99_in'], r['avg_out'], r['p99_out']]
            if has_turns:
                row_data.append(r['avg_turns'] or 'N/A')
            if has_cache:
                row_data.append(r['avg_cache'] or 'N/A')
            req_table.add_row(*row_data)

        dc.print('\n')
        dc.print(req_table)


class EmbeddingSummaryRenderer(BaseSummaryRenderer):
    """Renderer for Embedding / Rerank performance reports."""

    def _render_title(self, dc: DualConsole):
        dc.print(Panel(Text('Embedding/Rerank Performance Test Summary', style='bold'), width=80))

    def _get_token_stat_rows(self, result: AnalysisResult, args: Arguments) -> List[tuple]:
        total_tokens = result.total_tokens
        total_time = result.total_time
        return [
            ('Total Input Tokens', f'{total_tokens:,.0f} tokens'),
            ('Avg Input Rate', f'{total_tokens / total_time:.2f} tokens/sec' if total_time > 0 else 'N/A'),
        ]

    def _render_detail_table(self, result: AnalysisResult, dc: DualConsole):
        table = _make_detail_table()
        table.add_column('Conc.', justify='right', style='cyan')
        table.add_column('Rate', justify='right')
        table.add_column('RPS', justify='right')
        table.add_column('Avg Lat.(s)', justify='right')
        table.add_column('P99 Lat.(s)', justify='right')
        table.add_column('Avg Inp.TPS', justify='right')
        table.add_column('P99 Inp.TPS', justify='right')
        table.add_column('Avg Inp.Tok', justify='right')
        table.add_column('Success Rate', justify='right', style='green')

        self._fill_table_rows(table, result.rows, dc)
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
    if not result.rows:
        logger.warning('No available test result data to display')
        return

    console = Console(width=100)
    summary_file = os.path.join(args.outputs_dir, 'performance_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        file_console = Console(file=f, width=100, force_terminal=False)
        dc = DualConsole(console, file_console)
        renderer.render(result, args, dc)

    logger.info(f'Performance summary saved to: {summary_file}')
