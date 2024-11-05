import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Tuple

from evalscope.perf.utils._logging import logger


@dataclass
class BenchmarkData:
    request: Any = None
    start_time: float = field(default_factory=time.perf_counter)
    chunk_times: List[float] = field(default_factory=list)
    success: bool = False
    response_messages: List[Any] = field(default_factory=list)
    completed_time: float = 0.0

    def calculate_query_stream_metric(self) -> Tuple[float, int, float]:
        # firt chunk latency
        first_chunk_latency = self.chunk_times[0] - self.start_time
        n_chunks = len(self.chunk_times) - 2
        n_chunks_time = self.chunk_times[-2] - self.chunk_times[0]
        return first_chunk_latency, n_chunks, n_chunks_time


def get_result_db_path(args):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = './outputs'

    if args.name:
        result_db_path = os.path.join(output_dir, args.name)
    else:
        result_db_path = os.path.join(
            output_dir, f'{args.model}_benchmark_{current_time}.db')

    if not os.path.exists(os.path.dirname(result_db_path)):
        os.makedirs(os.path.dirname(result_db_path), exist_ok=True)

    logger.info('Save the result to : %s' % result_db_path)
    if os.path.exists(result_db_path):
        logger.warning('The db file exists, delete it and start again!.')
        sys.exit(1)

    return result_db_path


def summary_result(expected_number_of_queries, total_time, n_total_queries,
                   n_succeed_queries, n_failed_queries, qps, concurrency,
                   avg_latency, avg_first_chunk_latency, n_avg_chunks,
                   avg_chunk_time, avg_prompt_tokens, avg_completion_tokens,
                   avg_token_per_seconds, avg_time_per_token, result_db_path):
    logger.info('Benchmarking summary: ')
    logger.info(' Time taken for tests: %.3f seconds' % total_time)
    logger.info(' Expected number of requests: %s'
                % expected_number_of_queries)
    logger.info(' Number of concurrency: %d' % concurrency)
    logger.info(' Total requests: %d' % n_total_queries)
    logger.info(' Succeed requests: %d' % n_succeed_queries)
    logger.info(' Failed requests: %d' % n_failed_queries)
    logger.info(' Average QPS: %.3f' % qps)
    logger.info(' Average latency: %.3f' % avg_latency)
    logger.info(' Throughput(average output tokens per second): %.3f'
                % avg_token_per_seconds)
    logger.info(' Average time to first token: %.3f' % avg_first_chunk_latency)
    logger.info(' Average input tokens per request: %.3f' % avg_prompt_tokens)
    logger.info(' Average output tokens per request: %.3f'
                % avg_completion_tokens)
    logger.info(' Average time per output token: %.5f' % avg_time_per_token)
    logger.info(' Average package per request: %.3f' % n_avg_chunks)
    logger.info(' Average package latency: %.3f' % avg_chunk_time)

    con = sqlite3.connect(result_db_path)
    query_sql = (
        "SELECT start_time, chunk_times, success, completed_time, latency, first_chunk_latency, \
                   n_chunks, chunk_time, prompt_tokens, completion_tokens \
                   FROM result WHERE success='True' ORDER BY first_chunk_latency ASC"
    )

    percentiles = [50, 66, 75, 80, 90, 95, 98, 99]
    with con:
        rows = con.execute(query_sql).fetchall()
        n_success_queries = len(rows)
        if len(rows) > len(percentiles):
            logger.info(' Percentile of time to first token: ')
            for percentile in percentiles:
                idx = (int)(n_success_queries * percentile / 100)
                row = rows[idx]
                logger.info('     p%s: %.4f' %
                            (percentile,
                             row[5] if row[5] is not None else float('inf')))
            logger.info(' Percentile of request latency: ')
            latency_index = 4
            rows.sort(key=lambda x: x[latency_index])
            for percentile in percentiles:
                idx = (int)(n_success_queries * percentile / 100)
                row = rows[idx]
                logger.info('     p%s: %.4f' %
                            (percentile, row[latency_index] if
                             row[latency_index] is not None else float('inf')))
        else:
            logger.info(' Too little data to calculate quantiles!')
    con.close()
