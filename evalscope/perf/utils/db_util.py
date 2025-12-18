import base64
import json
import os
import pickle
import re
import sqlite3
import sys
from datetime import datetime
from tabulate import tabulate
from typing import Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics
from evalscope.utils.logger import get_logger

logger = get_logger()


class DatabaseColumns:
    REQUEST = 'request'
    START_TIME = 'start_time'
    INTER_TOKEN_LATENCIES = 'inter_token_latencies'
    SUCCESS = 'success'
    RESPONSE_MESSAGES = 'response_messages'
    COMPLETED_TIME = 'completed_time'
    LATENCY = 'latency'
    FIRST_CHUNK_LATENCY = 'first_chunk_latency'
    PROMPT_TOKENS = 'prompt_tokens'
    COMPLETION_TOKENS = 'completion_tokens'
    MAX_GPU_MEMORY_COST = 'max_gpu_memory_cost'
    TIME_PER_OUTPUT_TOKEN = 'time_per_output_token'


def load_prompt(prompt_path_or_text):
    if prompt_path_or_text.startswith('@'):
        with open(prompt_path_or_text[1:], 'r', encoding='utf-8') as file:
            return file.read()
    return prompt_path_or_text


def encode_data(data) -> str:
    """Encodes data using base64 and pickle."""
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')


def write_json_file(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def transpose_results(data):
    headers = data.keys()
    rows = zip(*data.values())

    return [dict(zip(headers, row)) for row in rows]


def create_result_table(cursor):
    cursor.execute(
        f'''CREATE TABLE IF NOT EXISTS result(
                      {DatabaseColumns.REQUEST} TEXT,
                      {DatabaseColumns.START_TIME} REAL,
                      {DatabaseColumns.INTER_TOKEN_LATENCIES} TEXT,
                      {DatabaseColumns.SUCCESS} INTEGER,
                      {DatabaseColumns.RESPONSE_MESSAGES} TEXT,
                      {DatabaseColumns.COMPLETED_TIME} REAL,
                      {DatabaseColumns.LATENCY} REAL,
                      {DatabaseColumns.FIRST_CHUNK_LATENCY} REAL,
                      {DatabaseColumns.PROMPT_TOKENS} INTEGER,
                      {DatabaseColumns.COMPLETION_TOKENS} INTEGER,
                      {DatabaseColumns.MAX_GPU_MEMORY_COST} REAL,
                      {DatabaseColumns.TIME_PER_OUTPUT_TOKEN} REAL
                   )'''
    )


def insert_benchmark_data(cursor: sqlite3.Cursor, benchmark_data: BenchmarkData):
    request = benchmark_data.request
    inter_token_latencies = json.dumps(benchmark_data.inter_chunk_latency)
    response_messages = encode_data(benchmark_data.response_messages)

    # Columns common to both success and failure cases
    common_columns = (
        request,
        benchmark_data.start_time,
        inter_token_latencies,
        benchmark_data.success,
        response_messages,
        benchmark_data.completed_time,
    )

    if benchmark_data.success:
        # Add additional columns for success case
        additional_columns = (
            benchmark_data.query_latency, benchmark_data.first_chunk_latency, benchmark_data.prompt_tokens,
            benchmark_data.completion_tokens, benchmark_data.max_gpu_memory_cost, benchmark_data.time_per_output_token
        )
        query = f"""INSERT INTO result(
                      {DatabaseColumns.REQUEST}, {DatabaseColumns.START_TIME}, {DatabaseColumns.INTER_TOKEN_LATENCIES},
                      {DatabaseColumns.SUCCESS}, {DatabaseColumns.RESPONSE_MESSAGES}, {DatabaseColumns.COMPLETED_TIME},
                      {DatabaseColumns.LATENCY}, {DatabaseColumns.FIRST_CHUNK_LATENCY}, {DatabaseColumns.PROMPT_TOKENS},
                      {DatabaseColumns.COMPLETION_TOKENS}, {DatabaseColumns.MAX_GPU_MEMORY_COST},
                      {DatabaseColumns.TIME_PER_OUTPUT_TOKEN}
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns + additional_columns)
    else:
        query = f"""INSERT INTO result(
                      {DatabaseColumns.REQUEST}, {DatabaseColumns.START_TIME}, {DatabaseColumns.INTER_TOKEN_LATENCIES},
                      {DatabaseColumns.SUCCESS}, {DatabaseColumns.RESPONSE_MESSAGES}, {DatabaseColumns.COMPLETED_TIME}
                   ) VALUES (?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns)


def get_output_path(args: Arguments) -> str:
    if args.no_timestamp:
        output_path = os.path.join(args.outputs_dir, f'{args.name or args.model_id}')
    else:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.outputs_dir, current_time, f'{args.name or args.model_id}')
    # Filter illegal characters
    output_path = re.sub(r'[<>:"|?*]', '_', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger.info(f'Save the result to: {output_path}')
    return output_path


def get_result_db_path(args: Arguments):
    result_db_path = os.path.join(args.outputs_dir, 'benchmark_data.db')

    logger.info(f'Save the data base to: {result_db_path}')
    if os.path.exists(result_db_path):
        logger.error(f'The db file {result_db_path} exists, delete it and start again!.')
        sys.exit(1)

    return result_db_path


class PercentileMetrics:
    TTFT = 'TTFT (s)'
    ITL = 'ITL (s)'
    TPOT = 'TPOT (s)'
    LATENCY = 'Latency (s)'
    INPUT_TOKENS = 'Input tokens'
    OUTPUT_TOKENS = 'Output tokens'
    OUTPUT_THROUGHPUT = 'Output (tok/s)'
    TOTAL_THROUGHPUT = 'Total (tok/s)'
    PERCENTILES = 'Percentiles'


def calculate_percentiles(data: List[float], percentiles: List[int]) -> Dict[int, float]:
    """
    Calculate the percentiles for a specific list of data.

    :param data: List of values for a specific metric.
    :param percentiles: List of percentiles to calculate.
    :return: Dictionary of calculated percentiles.
    """
    results = {}
    n_success_queries = len(data)
    data.sort()
    for percentile in percentiles:
        try:
            idx = int(n_success_queries * percentile / 100)
            value = data[idx] if data[idx] is not None else float('nan')
            results[percentile] = round(value, 4)
        except IndexError:
            results[percentile] = float('nan')
    return results


def get_percentile_results(result_db_path: str) -> Dict[str, List[float]]:
    """
    Compute and return quantiles for various metrics from the database results.

    :param result_db_path: Path to the SQLite database file.
    :return: Dictionary of percentiles for various metrics.
    """
    query_sql = f'''SELECT {DatabaseColumns.START_TIME}, {DatabaseColumns.INTER_TOKEN_LATENCIES}, {DatabaseColumns.SUCCESS},
                    {DatabaseColumns.COMPLETED_TIME}, {DatabaseColumns.LATENCY}, {DatabaseColumns.FIRST_CHUNK_LATENCY},
                    {DatabaseColumns.PROMPT_TOKENS},
                    {DatabaseColumns.COMPLETION_TOKENS}, {DatabaseColumns.TIME_PER_OUTPUT_TOKEN}
                    FROM result WHERE {DatabaseColumns.SUCCESS}=1'''  # noqa: E501

    percentiles = [10, 25, 50, 66, 75, 80, 90, 95, 98, 99]

    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        cursor.execute(query_sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

    # Create column index mapping
    col_indices = {col: idx for idx, col in enumerate(columns)}

    # Prepare data for each metric
    inter_token_latencies_all = []
    for row in rows:
        try:
            itl = json.loads(row[col_indices[DatabaseColumns.INTER_TOKEN_LATENCIES]]) or []
            inter_token_latencies_all.extend(itl)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f'Error parsing inter token latencies: {e}')

    metrics = {
        PercentileMetrics.TTFT: [row[col_indices[DatabaseColumns.FIRST_CHUNK_LATENCY]] for row in rows],
        PercentileMetrics.ITL:
        inter_token_latencies_all,
        PercentileMetrics.TPOT: [row[col_indices[DatabaseColumns.TIME_PER_OUTPUT_TOKEN]] for row in rows],
        PercentileMetrics.LATENCY: [row[col_indices[DatabaseColumns.LATENCY]] for row in rows],
        PercentileMetrics.INPUT_TOKENS: [row[col_indices[DatabaseColumns.PROMPT_TOKENS]] for row in rows],
        PercentileMetrics.OUTPUT_TOKENS: [row[col_indices[DatabaseColumns.COMPLETION_TOKENS]] for row in rows],
        PercentileMetrics.OUTPUT_THROUGHPUT:
        [(row[col_indices[DatabaseColumns.COMPLETION_TOKENS]] / row[col_indices[DatabaseColumns.LATENCY]])
         if row[col_indices[DatabaseColumns.LATENCY]] > 0 else float('nan') for row in rows],
        PercentileMetrics.TOTAL_THROUGHPUT:
        [((row[col_indices[DatabaseColumns.PROMPT_TOKENS]] + row[col_indices[DatabaseColumns.COMPLETION_TOKENS]])
          / row[col_indices[DatabaseColumns.LATENCY]])
         if row[col_indices[DatabaseColumns.LATENCY]] > 0 else float('nan') for row in rows]
    }

    # Calculate percentiles for each metric
    results = {PercentileMetrics.PERCENTILES: [f'{p}%' for p in percentiles]}
    for metric_name, data in metrics.items():
        metric_percentiles = calculate_percentiles(data, percentiles)
        results[metric_name] = [metric_percentiles[p] for p in percentiles]

    return results


def summary_result(args: Arguments, metrics: BenchmarkMetrics, result_db_path: str) -> Tuple[Dict, Dict]:
    result_path = os.path.dirname(result_db_path)
    write_json_file(args.to_dict(), os.path.join(result_path, 'benchmark_args.json'))

    metrics_result = metrics.create_message()
    write_json_file(metrics_result, os.path.join(result_path, 'benchmark_summary.json'))

    # Print summary in a table
    table = tabulate(list(metrics_result.items()), headers=['Key', 'Value'], tablefmt='grid')
    logger.info('\nBenchmarking summary:\n' + table)

    # Get percentile results
    percentile_result = get_percentile_results(result_db_path)
    if percentile_result:
        write_json_file(transpose_results(percentile_result), os.path.join(result_path, 'benchmark_percentile.json'))
        # Print percentile results in a table
        table = tabulate(percentile_result, headers='keys', tablefmt='pretty')
        logger.info('\nPercentile results:\n' + table)

    if args.dataset.startswith('speed_benchmark'):
        speed_benchmark_result(result_db_path)

    logger.info(f'Save the summary to: {result_path}')

    return metrics_result, percentile_result


def speed_benchmark_result(result_db_path: str):
    query_sql = f"""
        SELECT
            {DatabaseColumns.PROMPT_TOKENS},
            ROUND(AVG({DatabaseColumns.COMPLETION_TOKENS} / {DatabaseColumns.LATENCY}), 2) AS avg_completion_token_per_second,
            ROUND(AVG({DatabaseColumns.MAX_GPU_MEMORY_COST}), 2)
        FROM
            result
        WHERE
            {DatabaseColumns.SUCCESS} = 1 AND {DatabaseColumns.LATENCY} > 0
        GROUP BY
            {DatabaseColumns.PROMPT_TOKENS}
    """  # noqa: E501

    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        cursor.execute(query_sql)
        rows = cursor.fetchall()

    # Prepare data for tabulation
    headers = ['Prompt Tokens', 'Speed(tokens/s)', 'GPU Memory(GB)']
    data = [dict(zip(headers, row)) for row in rows]

    # Print results in a table
    table = tabulate(data, headers='keys', tablefmt='pretty')
    logger.info('\nSpeed Benchmark Results:\n' + table)

    # Write results to JSON file
    result_path = os.path.dirname(result_db_path)
    write_json_file(data, os.path.join(result_path, 'speed_benchmark.json'))


def average_results(results_list: List[Dict]):
    if not results_list:
        return {}

    avg_res = {'metrics': {}, 'percentiles': {}}

    # Metrics
    metric_keys = results_list[0]['metrics'].keys()
    for k in metric_keys:
        vals = [r['metrics'].get(k, 0) for r in results_list]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            avg_res['metrics'][k] = sum(vals) / len(vals)

    # Percentiles
    if results_list[0].get('percentiles'):
        perc_keys = results_list[0]['percentiles'].keys()
        for k in perc_keys:
            # Skip averaging for the labels list, just copy it
            if k == PercentileMetrics.PERCENTILES:
                avg_res['percentiles'][k] = results_list[0]['percentiles'][k]
                continue

            lists = [r['percentiles'][k] for r in results_list]
            avg_list = []
            if lists and lists[0]:
                length = len(lists[0])
                for i in range(length):
                    # Extract i-th percentile from all runs
                    col_vals = []
                    for l in lists:  # noqa: E741
                        if i < len(l) and isinstance(l[i], (int, float)):
                            col_vals.append(l[i])

                    if col_vals:
                        val = sum(col_vals) / len(col_vals)
                        avg_list.append(val)
                    else:
                        avg_list.append(0)  # Use 0 for N/A to maintain list structure
            avg_res['percentiles'][k] = avg_list

    # Copy other keys like 'percentiles_keys' if they exist
    if 'percentiles' in results_list[0]:
        for k, v in results_list[0]['percentiles'].items():
            if k not in avg_res['percentiles']:
                avg_res['percentiles'][k] = v

    return avg_res
