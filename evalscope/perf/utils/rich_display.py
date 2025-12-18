# the following code is largely adapted from https://github.com/lework/llm-benchmark

import numpy as np
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from evalscope.perf.arguments import Arguments
from evalscope.utils.logger import get_logger
from .benchmark_util import Metrics
from .db_util import PercentileMetrics

logger = get_logger()


def _print_to_both(console: Console, file_console: Console, content, **kwargs):
    """Helper to print to both console and file console."""
    console.print(content, **kwargs)
    # Remove style for file output
    file_kwargs = {k: v for k, v in kwargs.items() if k != 'style'}
    file_console.print(content, **file_kwargs)


def analyze_results(all_results):
    """Analyze all test results and generate a summary report"""
    summary = []
    total_tokens = 0
    total_time = 0

    # Handle both old list format and new dict format
    if isinstance(all_results, dict):
        # New format: {parallel_x_number_x: {metrics: ..., percentiles: ...}}
        results_list = []
        for key, value in all_results.items():
            if 'metrics' in value and 'percentiles' in value:
                results_list.append((value['metrics'], value['percentiles']))
        all_results = results_list

    for result in all_results:
        total_metrics = result[0]
        percentile_metrics = result[1]
        percentiles = percentile_metrics[PercentileMetrics.PERCENTILES]
        try:
            concurrency = total_metrics.get(Metrics.NUMBER_OF_CONCURRENCY, 0)
            rate = total_metrics.get(Metrics.REQUEST_RATE, 0)
            rps = total_metrics.get(Metrics.REQUEST_THROUGHPUT, 0)
            avg_latency = total_metrics.get(Metrics.AVERAGE_LATENCY, 0)
            p99_latency = percentile_metrics.get(PercentileMetrics.LATENCY)[percentiles.index('99%')]
            avg_tps = total_metrics.get(Metrics.OUTPUT_TOKEN_THROUGHPUT, 0)
            avg_ttft = total_metrics.get(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, 0)
            p99_ttft = percentile_metrics.get(PercentileMetrics.TTFT)[percentiles.index('99%')]
            success_rate = (
                total_metrics.get(Metrics.SUCCEED_REQUESTS, 0) / total_metrics.get(Metrics.TOTAL_REQUESTS, 1)
            ) * 100
            avg_tpot = total_metrics.get(Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN, 0)
            p99_tpot = percentile_metrics.get(PercentileMetrics.TPOT)[percentiles.index('99%')]

            # Ensure all values are valid numbers
            if any(x is None for x in [concurrency, rps, avg_latency, p99_latency, avg_tps, avg_ttft]):
                logger.warning(f'Warning: Test results for concurrency {concurrency} contain invalid data, skipped')
                continue

            summary.append([
                str(int(concurrency)),
                str(int(rate)) if rate != -1 else 'INF',
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
            logger.warning(
                f"Warning: Error processing results for concurrency {result.get('concurrency', 'unknown')}: {str(e)}"
            )
            continue

    if not summary:
        logger.warning('Error: No valid test result data')
        return [], 0, 0

    # Sort summary by concurrency and rate
    summary.sort(key=lambda x: (int(x[0]), int(x[1]) if x[1] != 'INF' else float('inf')))

    return summary, total_tokens, total_time


def print_summary(all_results, args: Arguments):
    """Print test results summary and save to file."""
    summary, total_tokens, total_time = analyze_results(all_results)

    if not summary:
        logger.warning('No available test result data to display')
        return

    console = Console(width=100)
    summary_file = os.path.join(args.outputs_dir, 'performance_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        file_console = Console(file=f, width=100, force_terminal=False)

        # Create title panel
        title = Text('Performance Test Summary Report', style='bold')
        _print_to_both(console, file_console, Panel(title, width=80))

        # Print basic information
        basic_info = Table(show_header=False, width=80)
        basic_info.add_column('Name', style='cyan', width=25)
        basic_info.add_column('Value', style='green', width=55)

        basic_info.add_row('Model', args.model_id)
        basic_info.add_row('Test Dataset', args.dataset)
        basic_info.add_row('Total Generated', f'{total_tokens:,} tokens')
        basic_info.add_row('Total Test Time', f'{total_time:.2f} seconds')
        basic_info.add_row('Avg Output Rate', f'{total_tokens / total_time:.2f} tokens/sec')
        basic_info.add_row('Output Path', args.outputs_dir)

        _print_to_both(console, file_console, '\nBasic Information:')
        _print_to_both(console, file_console, basic_info)

        # Create detailed performance metrics table
        table = Table(
            title='Detailed Performance Metrics',
            show_header=True,
            header_style='bold cyan',
            border_style='blue',
            pad_edge=False,
            expand=False,
        )

        # Add columns
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

        # Add data rows
        for row in summary:
            try:
                success_rate = float(row[-1].rstrip('%'))
                row_style = 'green' if success_rate >= 95 else 'yellow' if success_rate >= 80 else 'red'

                table.add_row(*row, style=row_style)
            except ValueError as e:
                error_msg = f'Warning: Error processing row data: {str(e)}'
                _print_to_both(console, file_console, error_msg, style='bold red')
                continue

        _print_to_both(console, file_console, '\n')
        _print_to_both(console, file_console, table)

        # Calculate and display best performance configuration
        try:
            best_rps_idx = np.argmax([float(row[2]) if row[2] != 'N/A' else -1 for row in summary])
            best_latency_idx = np.argmin([float(row[3]) if row[3] != 'N/A' else float('inf') for row in summary])

            perf_info = Table(title='Best Performance Configuration', show_header=False, box=None, width=60)
            perf_info.add_column('Metric', style='cyan', width=20)
            perf_info.add_column('Value', style='green', width=40)

            perf_info.add_row(
                'Highest RPS', f'Concurrency {summary[best_rps_idx][0]} ({summary[best_rps_idx][2]} req/sec)'
            )
            perf_info.add_row(
                'Lowest Latency', f'Concurrency {summary[best_latency_idx][0]} ({summary[best_latency_idx][3]} seconds)'
            )

            _print_to_both(console, file_console, '\n')
            _print_to_both(console, file_console, perf_info)

            # Performance recommendations
            recommendations = []
            if best_rps_idx == len(summary) - 1:
                recommendations.append(
                    'The system seems not to have reached its performance bottleneck, try higher concurrency'
                )
            elif best_rps_idx == 0:
                recommendations.append('Consider lowering concurrency, current load may be too high')
            else:
                recommendations.append(f'Optimal concurrency range is around {summary[best_rps_idx][0]}')

            success_rate = float(summary[-1][10][:-1])
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

    logger.info(f'Performance summary saved to: {summary_file}')
