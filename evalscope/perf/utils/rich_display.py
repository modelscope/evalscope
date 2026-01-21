import numpy as np
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from evalscope.perf.arguments import Arguments
from evalscope.utils.logger import get_logger
from .benchmark_util import Metrics, is_embedding_or_rerank_api
from .db_util import PercentileMetrics

logger = get_logger()


def _print_to_both(console: Console, file_console: Console, content, **kwargs):
    """Helper to print to both console and file console."""
    console.print(content, **kwargs)
    # Remove style for file output
    file_kwargs = {k: v for k, v in kwargs.items() if k != 'style'}
    file_console.print(content, **file_kwargs)


def _normalize_results(all_results):
    """Normalize input results from dict to list format if necessary."""
    if isinstance(all_results, dict):
        results_list = []
        for key, value in all_results.items():
            if 'metrics' in value and 'percentiles' in value:
                results_list.append((value['metrics'], value['percentiles']))
        return results_list
    return all_results


def _analyze_results_embedding(all_results):
    """Analyze results specifically for Embedding/Rerank models."""
    summary = []
    total_tokens = 0
    total_time = 0

    results = _normalize_results(all_results)

    for result in results:
        total_metrics = result[0]
        percentile_metrics = result[1]
        percentiles = percentile_metrics[PercentileMetrics.PERCENTILES]

        try:
            concurrency = total_metrics.get(Metrics.NUMBER_OF_CONCURRENCY, 0)
            rate = total_metrics.get(Metrics.REQUEST_RATE, 0)
            rps = total_metrics.get(Metrics.REQUEST_THROUGHPUT, 0)
            avg_latency = total_metrics.get(Metrics.AVERAGE_LATENCY, 0)
            p99_latency = percentile_metrics.get(PercentileMetrics.LATENCY)[percentiles.index('99%')]
            success_rate = (
                total_metrics.get(Metrics.SUCCEED_REQUESTS, 0) / total_metrics.get(Metrics.TOTAL_REQUESTS, 1)
            ) * 100

            # Embedding specific metrics
            avg_input_tps = total_metrics.get(Metrics.INPUT_TOKEN_THROUGHPUT, 0)
            p99_input_tps = percentile_metrics.get(PercentileMetrics.INPUT_THROUGHPUT,
                                                   [0] * len(percentiles))[percentiles.index('99%')]
            avg_input_tokens = total_metrics.get(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST, 0)

            # Validation
            if any(x is None for x in [concurrency, rps, avg_latency, p99_latency]):
                logger.warning(f'Warning: Test results for concurrency {concurrency} contain invalid data, skipped')
                continue

            summary.append([
                str(int(concurrency)),
                f'{rate:.2f}' if rate != -1 else 'INF',
                f'{rps:.2f}' if rps is not None else 'N/A',
                f'{avg_latency:.3f}' if avg_latency is not None else 'N/A',
                f'{p99_latency:.3f}' if p99_latency is not None else 'N/A',
                f'{avg_input_tps:.2f}' if avg_input_tps is not None else 'N/A',
                f'{p99_input_tps:.2f}' if p99_input_tps is not None else 'N/A',
                f'{avg_input_tokens:.1f}' if avg_input_tokens is not None else 'N/A',
                f'{success_rate:.1f}%' if success_rate is not None else 'N/A',
            ])

            total_tokens += total_metrics.get(Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST,
                                              0) * total_metrics.get(Metrics.SUCCEED_REQUESTS, 0)
            total_time += total_metrics.get(Metrics.TIME_TAKEN_FOR_TESTS, 0)

        except Exception as e:
            logger.warning(f'Warning: Error processing embedding results: {str(e)}')
            continue

    if summary:
        summary.sort(key=lambda x: (float(x[0]), float(x[1]) if x[1] != 'INF' else float('inf')))

    return summary, total_tokens, total_time


def _analyze_results_llm(all_results):
    """Analyze results specifically for LLM Generation models."""
    summary = []
    total_tokens = 0
    total_time = 0

    results = _normalize_results(all_results)

    for result in results:
        total_metrics = result[0]
        percentile_metrics = result[1]
        percentiles = percentile_metrics[PercentileMetrics.PERCENTILES]

        try:
            concurrency = total_metrics.get(Metrics.NUMBER_OF_CONCURRENCY, 0)
            rate = total_metrics.get(Metrics.REQUEST_RATE, 0)
            rps = total_metrics.get(Metrics.REQUEST_THROUGHPUT, 0)
            avg_latency = total_metrics.get(Metrics.AVERAGE_LATENCY, 0)
            p99_latency = percentile_metrics.get(PercentileMetrics.LATENCY)[percentiles.index('99%')]
            success_rate = (
                total_metrics.get(Metrics.SUCCEED_REQUESTS, 0) / total_metrics.get(Metrics.TOTAL_REQUESTS, 1)
            ) * 100

            # LLM specific metrics
            avg_tps = total_metrics.get(Metrics.OUTPUT_TOKEN_THROUGHPUT, 0)
            avg_ttft = total_metrics.get(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, 0)
            p99_ttft = percentile_metrics.get(PercentileMetrics.TTFT)[percentiles.index('99%')]
            avg_tpot = total_metrics.get(Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, 0)
            p99_tpot = percentile_metrics.get(PercentileMetrics.TPOT)[percentiles.index('99%')]

            # Validation
            if any(x is None for x in [concurrency, rps, avg_latency, p99_latency]):
                logger.warning(f'Warning: Test results for concurrency {concurrency} contain invalid data, skipped')
                continue

            summary.append([
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
            ])

            total_tokens += total_metrics.get(Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST,
                                              0) * total_metrics.get(Metrics.SUCCEED_REQUESTS, 0)
            total_time += total_metrics.get(Metrics.TIME_TAKEN_FOR_TESTS, 0)

        except Exception as e:
            logger.warning(f'Warning: Error processing LLM results: {str(e)}')
            continue

    if summary:
        summary.sort(key=lambda x: (float(x[0]), float(x[1]) if x[1] != 'INF' else float('inf')))

    return summary, total_tokens, total_time


def analyze_results(all_results, api_type: str = None):
    """Dispatcher for result analysis based on API type."""
    is_embedding_rerank = is_embedding_or_rerank_api(api_type)

    if is_embedding_rerank:
        summary, total_tokens, total_time = _analyze_results_embedding(all_results)
    else:
        summary, total_tokens, total_time = _analyze_results_llm(all_results)

    if not summary:
        logger.warning('Error: No valid test result data')
        return [], 0, 0, is_embedding_rerank

    return summary, total_tokens, total_time, is_embedding_rerank


def _generate_recommendations(summary, console, file_console):
    """Shared logic for generating performance recommendations based on summary table."""
    try:
        # Common indices for both modes: RPS is col 2, Latency is col 3
        best_rps_idx = np.argmax([float(row[2]) if row[2] != 'N/A' else -1 for row in summary])
        best_latency_idx = np.argmin([float(row[3]) if row[3] != 'N/A' else float('inf') for row in summary])

        perf_info = Table(title='Best Performance Configuration', show_header=False, box=None, width=60)
        perf_info.add_column('Metric', style='cyan', width=20)
        perf_info.add_column('Value', style='green', width=40)

        perf_info.add_row('Highest RPS', f'Concurrency {summary[best_rps_idx][0]} ({summary[best_rps_idx][2]} req/sec)')
        perf_info.add_row(
            'Lowest Latency', f'Concurrency {summary[best_latency_idx][0]} ({summary[best_latency_idx][3]} seconds)'
        )

        _print_to_both(console, file_console, '\n')
        _print_to_both(console, file_console, perf_info)

        recommendations = []
        if best_rps_idx == len(summary) - 1:
            recommendations.append(
                'The system seems not to have reached its performance bottleneck, try higher concurrency'
            )
        elif best_rps_idx == 0:
            recommendations.append('Consider lowering concurrency, current load may be too high')
        else:
            recommendations.append(f'Optimal concurrency range is around {summary[best_rps_idx][0]}')

        # Success rate is the last column in the row for both modes
        success_rate_str = summary[-1][-1].rstrip('%')
        success_rate = float(success_rate_str) if success_rate_str != 'N/A' else 0

        if success_rate < 95:
            recommendations.append(
                'Success rate is low at high concurrency, check system resources or reduce concurrency'
            )

        _print_to_both(console, file_console, '\nPerformance Recommendations:', style='bold cyan')

        for rec in recommendations:
            _print_to_both(console, file_console, f'• {rec}', style='yellow')

    except Exception as e:
        error_msg = f'Warning: Error generating performance analysis: {str(e)}'
        _print_to_both(console, file_console, error_msg, style='bold red')


def _print_summary_embedding(summary, total_tokens, total_time, args, console, file_console):
    """Print summary specifically for Embedding/Rerank models."""
    # 1. Title
    title = Text('Embedding/Rerank Performance Test Summary', style='bold')
    _print_to_both(console, file_console, Panel(title, width=80))

    # 2. Basic Info
    basic_info = Table(show_header=False, width=80)
    basic_info.add_column('Name', style='cyan', width=25)
    basic_info.add_column('Value', style='green', width=55)

    basic_info.add_row('Model', args.model_id)
    basic_info.add_row('Test Dataset', args.dataset)
    basic_info.add_row('API Type', args.api)
    basic_info.add_row('Total Input Tokens', f'{total_tokens:,.0f} tokens')
    basic_info.add_row('Total Test Time', f'{total_time:.2f} seconds')
    basic_info.add_row('Avg Input Rate', f'{total_tokens / total_time:.2f} tokens/sec' if total_time > 0 else 'N/A')
    basic_info.add_row('Output Path', args.outputs_dir)

    _print_to_both(console, file_console, '\nBasic Information:')
    _print_to_both(console, file_console, basic_info)

    # 3. Detailed Table
    table = Table(
        title='Detailed Performance Metrics',
        show_header=True,
        header_style='bold cyan',
        border_style='blue',
        pad_edge=False,
        expand=False,
    )
    table.add_column('Conc.', justify='right', style='cyan')
    table.add_column('Rate', justify='right')
    table.add_column('RPS', justify='right')
    table.add_column('Avg Lat.(s)', justify='right')
    table.add_column('P99 Lat.(s)', justify='right')
    table.add_column('Avg Inp.TPS', justify='right')
    table.add_column('P99 Inp.TPS', justify='right')
    table.add_column('Avg Inp.Tok', justify='right')
    table.add_column('Success Rate', justify='right', style='green')

    for row in summary:
        try:
            success_rate = float(row[-1].rstrip('%'))
            row_style = 'green' if success_rate >= 95 else 'yellow' if success_rate >= 80 else 'red'
            table.add_row(*row, style=row_style)
        except ValueError as e:
            _print_to_both(console, file_console, f'Warning: Error processing row: {e}', style='bold red')
            continue

    _print_to_both(console, file_console, '\n')
    _print_to_both(console, file_console, table)

    # 4. Recommendations
    _generate_recommendations(summary, console, file_console)


def _print_summary_llm(summary, total_tokens, total_time, args, console, file_console):
    """Print summary specifically for LLM Generation models."""
    # 1. Title
    title = Text('Performance Test Summary Report', style='bold')
    _print_to_both(console, file_console, Panel(title, width=80))

    # 2. Basic Info
    basic_info = Table(show_header=False, width=80)
    basic_info.add_column('Name', style='cyan', width=25)
    basic_info.add_column('Value', style='green', width=55)

    basic_info.add_row('Model', args.model_id)
    basic_info.add_row('Test Dataset', args.dataset)
    basic_info.add_row('API Type', args.api)
    basic_info.add_row('Total Generated', f'{total_tokens:,} tokens')
    basic_info.add_row('Total Test Time', f'{total_time:.2f} seconds')
    basic_info.add_row('Avg Output Rate', f'{total_tokens / total_time:.2f} tokens/sec' if total_time > 0 else 'N/A')
    basic_info.add_row('Output Path', args.outputs_dir)

    _print_to_both(console, file_console, '\nBasic Information:')
    _print_to_both(console, file_console, basic_info)

    # 3. Detailed Table
    table = Table(
        title='Detailed Performance Metrics',
        show_header=True,
        header_style='bold cyan',
        border_style='blue',
        pad_edge=False,
        expand=False,
    )
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

    for row in summary:
        try:
            success_rate = float(row[-1].rstrip('%'))
            row_style = 'green' if success_rate >= 95 else 'yellow' if success_rate >= 80 else 'red'
            table.add_row(*row, style=row_style)
        except ValueError as e:
            _print_to_both(console, file_console, f'Warning: Error processing row: {e}', style='bold red')
            continue

    _print_to_both(console, file_console, '\n')
    _print_to_both(console, file_console, table)

    # 4. Recommendations
    _generate_recommendations(summary, console, file_console)


def print_summary(all_results, args: Arguments):
    """Print test results summary and save to file."""
    # Analyze results based on API type
    summary, total_tokens, total_time, is_embedding_rerank = analyze_results(all_results, api_type=args.api)

    if not summary:
        logger.warning('No available test result data to display')
        return

    console = Console(width=100)
    summary_file = os.path.join(args.outputs_dir, 'performance_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        file_console = Console(file=f, width=100, force_terminal=False)

        # Dispatch to specific print function
        if is_embedding_rerank:
            _print_summary_embedding(summary, total_tokens, total_time, args, console, file_console)
        else:
            _print_summary_llm(summary, total_tokens, total_time, args, console, file_console)

    logger.info(f'Performance summary saved to: {summary_file}')
