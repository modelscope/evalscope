import base64
import os
import pickle
import sqlite3
import sys
from datetime import datetime

import json
from tabulate import tabulate

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics
from evalscope.utils.logger import get_logger

logger = get_logger()


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
    cursor.execute('''CREATE TABLE IF NOT EXISTS result(
                      request TEXT,
                      start_time REAL,
                      chunk_times TEXT,
                      success INTEGER,
                      response_messages TEXT,
                      completed_time REAL,
                      latency REAL,
                      first_chunk_latency REAL,
                      n_chunks INTEGER,
                      chunk_time REAL,
                      prompt_tokens INTEGER,
                      completion_tokens INTEGER,
                      max_gpu_memory_cost REAL)''')


def insert_benchmark_data(cursor: sqlite3.Cursor, benchmark_data: BenchmarkData):
    request = encode_data(benchmark_data.request)
    chunk_times = json.dumps(benchmark_data.chunk_times)
    response_messages = encode_data(benchmark_data.response_messages)

    # Columns common to both success and failure cases
    common_columns = (
        request,
        benchmark_data.start_time,
        chunk_times,
        benchmark_data.success,
        response_messages,
        benchmark_data.completed_time,
    )

    if benchmark_data.success:
        # Add additional columns for success case
        additional_columns = (
            benchmark_data.query_latency,
            benchmark_data.first_chunk_latency,
            benchmark_data.n_chunks,
            benchmark_data.n_chunks_time,
            benchmark_data.prompt_tokens,
            benchmark_data.completion_tokens,
            benchmark_data.max_gpu_memory_cost,
        )
        query = """INSERT INTO result(
                      request, start_time, chunk_times, success, response_messages,
                      completed_time, latency, first_chunk_latency,
                      n_chunks, chunk_time, prompt_tokens, completion_tokens, max_gpu_memory_cost
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns + additional_columns)
    else:
        query = """INSERT INTO result(
                      request, start_time, chunk_times, success, response_messages, completed_time
                   ) VALUES (?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns)


def get_result_db_path(name, model):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = './outputs'
    result_db_path = os.path.join(output_dir, f'{name or model}_perf', current_time, 'benchmark_data.db')

    if not os.path.exists(os.path.dirname(result_db_path)):
        os.makedirs(os.path.dirname(result_db_path), exist_ok=True)

    logger.info(f'Save the result to: {result_db_path}')
    if os.path.exists(result_db_path):
        logger.warning('The db file exists, delete it and start again!.')
        sys.exit(1)

    return result_db_path


def get_percentile_results(result_db_path: str):

    def percentile_results(rows, index, percentiles):
        results = {}
        n_success_queries = len(rows)
        for percentile in percentiles:
            idx = int(n_success_queries * percentile / 100)
            row = rows[idx]
            value = row[index] if row[index] is not None else float('inf')
            results[percentile] = round(value, 4)
        return results

    query_sql = ('SELECT start_time, chunk_times, success, completed_time, latency, first_chunk_latency, '
                 'n_chunks, chunk_time, prompt_tokens, completion_tokens '
                 'FROM result WHERE success=1 ORDER BY first_chunk_latency ASC')
    percentiles = [10, 25, 50, 66, 75, 80, 90, 95, 98, 99]

    with sqlite3.connect(result_db_path) as con:
        rows = con.execute(query_sql).fetchall()

    if len(rows) <= len(percentiles):
        logger.info('Too little data to calculate quantiles!')
        return {}

    # Calculate percentiles for first chunk latency and latency
    first_chunk_latency_index = 5
    latency_index = 4

    first_chunk_latency_results = percentile_results(rows, first_chunk_latency_index, percentiles)
    rows.sort(key=lambda x: x[latency_index])
    latency_results = percentile_results(rows, latency_index, percentiles)

    # Prepare data for tabulation
    return {
        'Percentile': [f'{p}%' for p in percentiles],
        'First Chunk Latency (s)': [first_chunk_latency_results[p] for p in percentiles],
        'Latency (s)': [latency_results[p] for p in percentiles]
    }


def summary_result(args: Arguments, metrics: BenchmarkMetrics, expected_number_of_queries: int, result_db_path: str):
    result_path = os.path.dirname(result_db_path)
    write_json_file(args.to_dict(), os.path.join(result_path, 'benchmark_args.json'))

    data = metrics.create_message()
    data.update({'Expected number of requests': expected_number_of_queries, 'Result DB path': result_db_path})
    write_json_file(data, os.path.join(result_path, 'benchmark_summary.json'))

    # Print summary in a table
    table = tabulate(list(data.items()), headers=['Key', 'Value'], tablefmt='grid')
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


def speed_benchmark_result(result_db_path: str):
    query_sql = """
        SELECT
            prompt_tokens,
            ROUND(AVG(completion_tokens / latency), 2) AS avg_completion_token_per_second,
            ROUND(AVG(max_gpu_memory_cost), 2)
        FROM
            result
        WHERE
            success = 1 AND latency > 0
        GROUP BY
            prompt_tokens
    """

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
