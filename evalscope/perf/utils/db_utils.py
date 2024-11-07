import base64
import os
import pickle
import sqlite3
import sys
from datetime import datetime

import json
from tabulate import tabulate

from evalscope.perf.utils._logging import logger
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics


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
                      completion_tokens INTEGER)''')


def insert_benchmark_data(cursor: sqlite3.Cursor, benchmark_data: BenchmarkData):
    request = base64.b64encode(pickle.dumps(benchmark_data.request)).decode('utf-8')
    chunk_times = json.dumps(benchmark_data.chunk_times)
    response_messages = base64.b64encode(pickle.dumps(benchmark_data.response_messages)).decode('utf-8')

    if benchmark_data.success:
        latency = benchmark_data.query_latency
        n_chunks = benchmark_data.n_chunks
        first_chunk_latency = benchmark_data.first_chunk_latency
        chunk_time = benchmark_data.n_chunks_time

        cursor.execute(
            '''INSERT INTO result(request, start_time, chunk_times, success, response_messages,
                          completed_time, latency, first_chunk_latency, n_chunks, chunk_time,
                          prompt_tokens, completion_tokens)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (request, benchmark_data.start_time, chunk_times, benchmark_data.success, response_messages,
             benchmark_data.completed_time, latency, first_chunk_latency, n_chunks, chunk_time,
             benchmark_data.query_prompt_tokens, benchmark_data.query_completion_tokens))
    else:
        cursor.execute(
            '''INSERT INTO result(request, start_time, chunk_times, success, response_messages, completed_time)
                          VALUES (?, ?, ?, ?, ?, ?)''',
            (request, benchmark_data.start_time, chunk_times, benchmark_data.success, response_messages,
             benchmark_data.completed_time))


def get_result_db_path(name, model):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = './outputs'
    result_db_path = os.path.join(output_dir, f'{name or model}_benchmark_{current_time}.db')

    if not os.path.exists(os.path.dirname(result_db_path)):
        os.makedirs(os.path.dirname(result_db_path), exist_ok=True)

    logger.info(f'Save the result to: {result_db_path}')
    if os.path.exists(result_db_path):
        logger.warning('The db file exists, delete it and start again!.')
        sys.exit(1)

    return result_db_path


def get_percentile_results(result_db_path):

    def percentile_results(rows, n_success_queries, percentiles, index):
        results = {}
        for percentile in percentiles:
            idx = int(n_success_queries * percentile / 100)
            row = rows[idx]
            value = row[index] if row[index] is not None else float('inf')
            results[percentile] = round(value, 4)
        return results

    con = sqlite3.connect(result_db_path)
    query_sql = ('SELECT start_time, chunk_times, success, completed_time, latency, first_chunk_latency, '
                 'n_chunks, chunk_time, prompt_tokens, completion_tokens '
                 'FROM result WHERE success=1 ORDER BY first_chunk_latency ASC')
    tabulate_data = {}
    percentiles = [10, 25, 50, 66, 75, 80, 90, 95, 98, 99]
    with con:
        rows = con.execute(query_sql).fetchall()
        n_success_queries = len(rows)
        if n_success_queries > len(percentiles):
            # Calculate first chunk latency percentiles
            first_chunk_latency_index = 5
            first_chunk_latency_results = \
                percentile_results(rows, n_success_queries, percentiles, first_chunk_latency_index)

            # Sort rows by latency and calculate latency percentiles
            latency_index = 4
            rows.sort(key=lambda x: x[latency_index])
            latency_results = percentile_results(rows, n_success_queries, percentiles, latency_index)

            # Prepare data for tabulate
            tabulate_data = {
                'Percentile': [f'{p}%' for p in percentiles],
                'First Chunk Latency': [first_chunk_latency_results[p] for p in percentiles],
                'Latency': [latency_results[p] for p in percentiles]
            }
    return tabulate_data


def summary_result(metrics: BenchmarkMetrics, expected_number_of_queries: int, result_db_path: str):
    data = metrics.create_message()
    data.update({'Expected number of requests': expected_number_of_queries, 'Result DB path': result_db_path})
    data = [(key, value) for key, value in data.items()]
    table = tabulate(data, headers=['key', 'Value'], tablefmt='grid')
    logger.info('\nBenchmarking summary: \n' + table)

    percentile_result = get_percentile_results(result_db_path)
    if percentile_result:
        table = tabulate(percentile_result, headers='keys', tablefmt='pretty')
        logger.info('\nPercentile results: \n' + table)
    else:
        logger.info('Too little data to calculate quantiles!')
