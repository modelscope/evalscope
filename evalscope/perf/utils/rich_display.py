import numpy as np
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import Any, Dict, List, NamedTuple

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
        rows:         Formatted summary rows as a list of dicts keyed by semantic
                      column names (e.g. ``'concurrency'``, ``'rps'``,
                      ``'avg_latency'``, ``'success_rate'``).  Dict insertion
                      order matches the rendered table column order.
        total_tokens: Aggregate token count across all concurrency levels.
        total_time:   Aggregate test duration (seconds).
        all_results:  Original raw results retained for extra rendering
                      (e.g. the LLM Request Metrics table).
    """
    rows: List[Dict[str, str]]
    total_tokens: float
    total_time: float
    all_results: Any = None


class ColSpec(NamedTuple):
    """Unified column specification tying a row-dict key to a Rich table header.

    Using ``ColSpec`` as the single source of truth means the analyzer (which
    writes the row dict) and the renderer (which declares Rich columns) both
    reference the same object — a key rename propagates automatically.

    Attributes:
        key:     Row-dict key used by analyzers when building :class:`AnalysisResult` rows.
        header:  Column header displayed in the Rich table.
        style:   Optional Rich style applied to the column (e.g. ``'cyan'``).
        justify: Rich column justification (default ``'right'``).
    """
    key: str
    header: str
    style: str = ''
    justify: str = 'right'


class LLMCol:
    """
    Column specs for the LLM Detailed Performance Metrics table.

    ``ALL`` is the ordered list consumed by both the analyzer (row-dict keys)
    and the renderer (Rich column definitions).
    """
    CONCURRENCY = ColSpec('concurrency', 'Conc.', style='cyan')
    RATE = ColSpec('rate', 'Rate')
    NUM = ColSpec('num', 'Num')
    RPS = ColSpec('rps', 'RPS')
    AVG_LATENCY = ColSpec('avg_latency', 'Avg\nLat.(s)')
    P99_LATENCY = ColSpec('p99_latency', 'P99\nLat.(s)')
    AVG_TTFT = ColSpec('avg_ttft', 'Avg\nTTFT(ms)')
    P99_TTFT = ColSpec('p99_ttft', 'P99\nTTFT(ms)')
    AVG_TPOT = ColSpec('avg_tpot', 'Avg\nTPOT(ms)')
    P99_TPOT = ColSpec('p99_tpot', 'P99\nTPOT(ms)')
    AVG_TPS = ColSpec('avg_tps', 'Gen.\ntoks/s')
    SUCCESS_RATE = ColSpec('success_rate', 'Success\nRate', style='green')

    ALL: List[ColSpec] = [
        CONCURRENCY,
        RATE,
        NUM,
        RPS,
        AVG_LATENCY,
        P99_LATENCY,
        AVG_TTFT,
        P99_TTFT,
        AVG_TPOT,
        P99_TPOT,
        AVG_TPS,
        SUCCESS_RATE,
    ]


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


class ReqMetCol:
    """Column specs for the LLM Request Metrics table (per-request token / turn stats).

    ``FIXED`` columns are always rendered; optional columns appear only when
    non-trivial data is present.
    """
    # Always shown
    CONCURRENCY = ColSpec('concurrency', 'Conc.', style='cyan')
    NUM = ColSpec('num', 'Num')
    AVG_IN = ColSpec('avg_in', 'Avg In\nToks')
    P99_IN = ColSpec('p99_in', 'P99 In\nToks')
    AVG_OUT = ColSpec('avg_out', 'Avg Out\nToks')
    P99_OUT = ColSpec('p99_out', 'P99 Out\nToks')
    FIXED: List[ColSpec] = [CONCURRENCY, NUM, AVG_IN, P99_IN, AVG_OUT, P99_OUT]

    # Conditionally shown
    AVG_TURNS = ColSpec('avg_turns', 'Avg\nTurns/Req')
    AVG_CACHE = ColSpec('avg_cache', 'Approx\nCache Hit', style='green')
    AVG_DECODED = ColSpec('avg_decoded', 'Decoded\nTok/Iter')
    SPEC_RATE = ColSpec('spec_rate', 'Spec.\nAccept Rate', style='cyan')


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
        rows.sort(
            key=lambda x: (
                float(x[LLMCol.CONCURRENCY.key]),
                float(x[LLMCol.RATE.key]) if x[LLMCol.RATE.key] != 'INF' else float('inf'),
            )
        )

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
        num_reqs = total_metrics.get(Metrics.TOTAL_REQUESTS, 0)
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
        # Convert TTFT and TPOT from seconds to milliseconds for display
        avg_ttft_ms = avg_ttft * 1000 if avg_ttft is not None else None
        p99_ttft_ms = p99_ttft * 1000 if p99_ttft is not None else None
        avg_tpot_ms = avg_tpot * 1000 if avg_tpot is not None else None
        p99_tpot_ms = p99_tpot * 1000 if p99_tpot is not None else None

        if any(x is None for x in [concurrency, rps, avg_latency, p99_latency]):
            raise ValueError(f'Test results for concurrency {concurrency} contain invalid data, skipped')

        row = {
            LLMCol.CONCURRENCY.key: 'INF' if concurrency == -1 else str(int(concurrency)),
            LLMCol.RATE.key: f'{rate:.2f}' if rate != -1 else 'INF',
            LLMCol.NUM.key: str(int(num_reqs)),
            LLMCol.RPS.key: f'{rps:.2f}' if rps is not None else 'N/A',
            LLMCol.AVG_LATENCY.key: f'{avg_latency:.3f}' if avg_latency is not None else 'N/A',
            LLMCol.P99_LATENCY.key: f'{p99_latency:.3f}' if p99_latency is not None else 'N/A',
            LLMCol.AVG_TTFT.key: f'{avg_ttft_ms:.1f}' if avg_ttft_ms is not None else 'N/A',
            LLMCol.P99_TTFT.key: f'{p99_ttft_ms:.1f}' if p99_ttft_ms is not None else 'N/A',
            LLMCol.AVG_TPOT.key: f'{avg_tpot_ms:.1f}' if avg_tpot_ms is not None else 'N/A',
            LLMCol.P99_TPOT.key: f'{p99_tpot_ms:.1f}' if p99_tpot_ms is not None else 'N/A',
            LLMCol.AVG_TPS.key: f'{avg_tps:.2f}' if avg_tps is not None else 'N/A',
            LLMCol.SUCCESS_RATE.key: f'{success_rate:.1f}%' if success_rate is not None else 'N/A',
        }
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

        row = {
            EmbCol.CONCURRENCY.key: 'INF' if concurrency == -1 else str(int(concurrency)),
            EmbCol.RATE.key: f'{rate:.2f}' if rate != -1 else 'INF',
            EmbCol.RPS.key: f'{rps:.2f}' if rps is not None else 'N/A',
            EmbCol.AVG_LATENCY.key: f'{avg_latency:.3f}' if avg_latency is not None else 'N/A',
            EmbCol.P99_LATENCY.key: f'{p99_latency:.3f}' if p99_latency is not None else 'N/A',
            EmbCol.AVG_INPUT_TPS.key: f'{avg_input_tps:.2f}' if avg_input_tps is not None else 'N/A',
            EmbCol.P99_INPUT_TPS.key: f'{p99_input_tps:.2f}',
            EmbCol.AVG_INPUT_TOKENS.key: f'{avg_input_tokens:.1f}' if avg_input_tokens is not None else 'N/A',
            EmbCol.SUCCESS_RATE.key: f'{success_rate:.1f}%' if success_rate is not None else 'N/A',
        }
        tokens = (
            total_metrics.get(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, 0)
            * total_metrics.get(Metrics.SUCCEED_REQUESTS, 0)
        )
        elapsed = total_metrics.get(Metrics.TIME_TAKEN_FOR_TESTS, 0)
        return row, tokens, elapsed


# ---------------------------------------------------------------------------
# Layer 3: Renderers  (Template Method pattern)
# ---------------------------------------------------------------------------


class BaseSummaryRenderer(ABC):
    """Renders a performance summary report to a :class:`DualConsole`.

    Uses the **Template Method** pattern.  The fixed rendering order is:

    1. Title panel
    2. Basic information table
    3. Detailed per-concurrency metrics table
    4. Extra tables  *(hook — default is no-op)*
    5. Best-configuration summary + recommendations
    """

    @staticmethod
    def _make_detail_table() -> Table:
        """Create a Rich Table with standard styling for Detailed Performance Metrics."""
        return Table(
            title='Detailed Performance Metrics',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )

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
    def _detail_columns(self) -> List[ColSpec]:
        """Return the ordered :class:`ColSpec` list for the Detailed Performance Metrics table.

        The list drives both Rich column creation and, implicitly, the row-dict
        key order (via :attr:`ColSpec.key`).
        """

    # ------------------------------------------------------------------
    # Concrete / shared steps

    def _render_detail_table(self, result: AnalysisResult, dc: DualConsole):
        """Build and render the detail table from :meth:`_detail_columns`.

        Subclasses no longer need to override this method — only
        :meth:`_detail_columns` needs to be implemented.
        """
        table = self._make_detail_table()
        for col in self._detail_columns():
            table.add_column(col.header, justify=col.justify, style=col.style)
        self._fill_table_rows(table, result.rows, dc)
        dc.print('\n')
        dc.print(table)

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
            best_rps_idx = np.argmax([float(r[LLMCol.RPS.key]) if r[LLMCol.RPS.key] != 'N/A' else -1 for r in rows])
            best_latency_idx = np.argmin([
                float(r[LLMCol.AVG_LATENCY.key]) if r[LLMCol.AVG_LATENCY.key] != 'N/A' else float('inf') for r in rows
            ])

            perf_info = Table(title='Best Performance Configuration', show_header=False, box=None, width=60)
            perf_info.add_column('Metric', style='cyan', width=20)
            perf_info.add_column('Value', style='green', width=40)
            perf_info.add_row(
                'Highest RPS',
                f'Concurrency {rows[best_rps_idx][LLMCol.CONCURRENCY.key]}'
                f' ({rows[best_rps_idx][LLMCol.RPS.key]} req/sec)',
            )
            perf_info.add_row(
                'Lowest Latency',
                f'Concurrency {rows[best_latency_idx][LLMCol.CONCURRENCY.key]}'
                f' ({rows[best_latency_idx][LLMCol.AVG_LATENCY.key]} seconds)',
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
                recommendations.append(
                    f'Optimal concurrency range is around {rows[best_rps_idx][LLMCol.CONCURRENCY.key]}'
                )

            success_rate_str = rows[-1][LLMCol.SUCCESS_RATE.key].rstrip('%')
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

    def _fill_table_rows(self, table: Table, rows: List[Dict[str, str]], dc: DualConsole):
        """Add *rows* to *table* with success-rate-based row colouring.

        Dict insertion order must match the table's column order (guaranteed
        when ``_process_one`` builds the dict in :attr:`ColSpec` key order).
        """
        for row in rows:
            try:
                success_rate = float(row[LLMCol.SUCCESS_RATE.key].rstrip('%'))
                row_style = 'green' if success_rate >= 95 else 'yellow' if success_rate >= 80 else 'red'
                table.add_row(*row.values(), style=row_style)
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

    def _detail_columns(self) -> List[ColSpec]:
        return LLMCol.ALL

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
        has_spec = False

        for entry in results:
            total_metrics, percentile_metrics = entry
            percentiles = percentile_metrics.get(PercentileMetrics.PERCENTILES, [])

            concurrency = total_metrics.get(Metrics.NUMBER_OF_CONCURRENCY, 0)
            num_reqs = total_metrics.get(Metrics.TOTAL_REQUESTS, 0)
            avg_in = total_metrics.get(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST)
            avg_out = total_metrics.get(Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST)
            avg_turns = total_metrics.get(Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST)
            avg_cache = total_metrics.get(Metrics.AVERAGE_CACHED_PERCENT)
            avg_decoded = total_metrics.get(Metrics.AVERAGE_DECODED_TOKENS_PER_ITER)
            avg_spec_rate = total_metrics.get(Metrics.APPROX_SPECULATIVE_ACCEPTANCE_RATE)

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
            if avg_decoded is not None and avg_decoded > 0:
                has_spec = True

            rows.append({
                ReqMetCol.CONCURRENCY.key: 'INF' if concurrency == -1 else str(int(concurrency)),
                ReqMetCol.NUM.key: str(int(num_reqs)),
                ReqMetCol.AVG_IN.key: (f'{avg_in:.1f}' if avg_in is not None else 'N/A'),
                ReqMetCol.P99_IN.key: (f'{p99_in:.1f}' if p99_in is not None else 'N/A'),
                ReqMetCol.AVG_OUT.key: (f'{avg_out:.1f}' if avg_out is not None else 'N/A'),
                ReqMetCol.P99_OUT.key: (f'{p99_out:.1f}' if p99_out is not None else 'N/A'),
                ReqMetCol.AVG_TURNS.key: (f'{avg_turns:.2f}' if (avg_turns is not None and avg_turns > 0) else None),
                ReqMetCol.AVG_CACHE.key: (f'{avg_cache:.1f}%' if (avg_cache is not None and avg_cache > 0) else None),
                ReqMetCol.AVG_DECODED.key: (
                    f'{avg_decoded:.2f}' if (avg_decoded is not None and avg_decoded > 0) else None
                ),
                ReqMetCol.SPEC_RATE.key: (f'{avg_spec_rate * 100:.1f}%' if avg_spec_rate is not None else None),
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
        for col in ReqMetCol.FIXED:
            req_table.add_column(col.header, justify=col.justify, style=col.style)
        if has_turns:
            c = ReqMetCol.AVG_TURNS
            req_table.add_column(c.header, justify=c.justify, style=c.style)
        if has_cache:
            c = ReqMetCol.AVG_CACHE
            req_table.add_column(c.header, justify=c.justify, style=c.style)
        if has_spec:
            for c in (ReqMetCol.AVG_DECODED, ReqMetCol.SPEC_RATE):
                req_table.add_column(c.header, justify=c.justify, style=c.style)

        for r in rows:
            row_data = [r[c.key] for c in ReqMetCol.FIXED]
            if has_turns:
                row_data.append(r[ReqMetCol.AVG_TURNS.key] or 'N/A')
            if has_cache:
                row_data.append(r[ReqMetCol.AVG_CACHE.key] or 'N/A')
            if has_spec:
                row_data.append(r[ReqMetCol.AVG_DECODED.key] or 'N/A')
                row_data.append(r[ReqMetCol.SPEC_RATE.key] or 'N/A')
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

    def _detail_columns(self) -> List[ColSpec]:
        return EmbCol.ALL


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

    console = Console(width=120)  # wide enough for 12 folded-header LLM columns
    summary_file = os.path.join(args.outputs_dir, 'performance_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        file_console = Console(file=f, width=120, force_terminal=False)
        dc = DualConsole(console, file_console)
        renderer.render(result, args, dc)

    logger.info(f'Performance summary saved to: {summary_file}')
