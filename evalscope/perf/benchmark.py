import asyncio
import base64
import functools
import os
import pickle
import platform
import signal
import sqlite3
import sys
import time
from http import HTTPStatus
from typing import List

import json
import numpy as np

from evalscope.perf.arguments import QueryParameters
from evalscope.perf.http_client import AioHttpClient
from evalscope.perf.plugin.registry import api_registry, dataset_registry
from evalscope.perf.utils._logging import logger
from evalscope.perf.utils.benchmark_util import (BenchmarkData,
                                                 get_result_db_path,
                                                 summary_result)
from evalscope.perf.utils.signal_handler import signal_handler

_query_send_completed = False
_data_process_completed = False


async def dispatch_requests_worker(request_queue: asyncio.Queue, args):
    query_generator_class = api_registry(args.api)
    if not query_generator_class:
        logger.info('Can not find query generator: %s' % args.api)
    query_generator = query_generator_class(args.tokenizer_path)
    total_query_counter = 0
    query_parameters = QueryParameters(args)
    if args.prompt is not None:
        if args.prompt.startswith('@'):
            with open(args.prompt, 'r', encoding='utf-8') as f:
                prompt = f.read()
        else:
            prompt = args.prompt
        messages = [{'role': 'user', 'content': prompt}]
        request = query_generator.build_request(messages, query_parameters)
        if args.number is None:
            await request_queue.put(request)
        else:
            for i in range(args.number):
                if args.rate == -1:
                    await request_queue.put(request)
                else:
                    interval = np.random.exponential(1.0 / args.rate)
                    await asyncio.sleep(interval)
                    await request_queue.put(request)
    elif args.dataset_path is not None:
        while True:
            message_generator_class = dataset_registry(args.dataset)
            if not message_generator_class:
                logger.info('Can not find dataset: %s plugin.' %
                            (args.dataset))
                sys.exit(1)
            message_generator = message_generator_class(query_parameters)
            for messages in message_generator:
                request = query_generator.build_request(
                    messages, query_parameters)
                if request is None:
                    continue
                await request_queue.put(request)
                total_query_counter += 1
                if args.number is not None:
                    if total_query_counter >= args.number:
                        break
                if args.rate == -1:
                    continue
                interval = np.random.exponential(1.0 / args.rate)
                await asyncio.sleep(interval)
            if args.number is None:
                break
            elif total_query_counter >= args.number:
                break
    else:
        raise Exception('Prompt or dataset is required!')
    return total_query_counter


async def send_requests_worker(task_id, request_queue: asyncio.Queue,
                               benchmark_data_queue: asyncio.Queue, args):
    client = AioHttpClient(
        args.url,
        conn_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        headers=args.headers,
        debug=args.debug,
    )
    async with client:
        while True:
            try:
                request = request_queue.get_nowait()
                request_queue.task_done()
            except asyncio.QueueEmpty:
                if _query_send_completed:
                    break
                await asyncio.sleep(0.01)
                continue
            # auto get start_time when initializing
            benchmark_data = BenchmarkData(request=request)
            collected_messages = []
            try:
                async for is_error, state_code, response_data in client.post(
                        request):
                    if is_error or state_code != HTTPStatus.OK:
                        logger.error(
                            'Request: %s failed, state_code: %s, data: %s' %
                            (request, state_code, response_data))
                        break
                    else:
                        if response_data:
                            collected_messages.append(response_data)
                            logger.info(response_data)
                            benchmark_data.chunk_times.append(
                                time.perf_counter())

                benchmark_data.response_messages = collected_messages
                benchmark_data.completed_time = time.perf_counter()
                benchmark_data.success = not is_error
                await benchmark_data_queue.put(benchmark_data)
            except BaseException as e:
                logger.exception(e)
                if response_data:
                    collected_messages.append(response_data)
                benchmark_data.response_messages = collected_messages
                benchmark_data.completed_time = time.perf_counter()
                await benchmark_data_queue.put(benchmark_data)
                logger.error('Request query: %s exception, response: %s' %
                             (request, response_data))


async def statistic_benchmark_metric_worker(
        benchmark_data_queue: asyncio.Queue, args):
    n_succeed_queries = 0
    n_failed_queries = 0
    total_first_chunk_latency = 0
    total_latency = 0.0
    n_total_chunks = 0
    n_total_prompt_tokens = 0
    n_total_completion_tokens = 0
    qps = 0
    concurrency = args.parallel
    start_time = None
    total_chunks_time = 0.0
    avg_latency = -1
    avg_first_chunk_latency = -1
    avg_token_per_seconds = -1
    avg_time_per_token = -1
    n_avg_chunks = -1
    avg_chunk_time = -1
    avg_prompt_tokens = -1
    avg_completion_tokens = -1
    total_time = 1
    n_total_queries = 0
    n_benchmark_result = 0
    api_plugin_class = api_registry(args.api)
    if not api_plugin_class:
        logger.info('Can not find query generator: %s' % args.api)
    api_plugin = api_plugin_class(args.tokenizer_path)

    result_db_path = get_result_db_path(args)

    con = sqlite3.connect(result_db_path)

    db_cur = con.cursor()
    db_cur.execute(
        'CREATE TABLE result(request, start_time, chunk_times, success, \
                   response_messages, completed_time, latency, first_chunk_latency, \
                   n_chunks, chunk_time, prompt_tokens, completion_tokens)')

    if args.wandb_api_key is not None:
        import wandb
        import datetime

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = (
            args.name if args.name is not None else '%s_%s' %
            (args.model, current_time))
        wandb.init(
            project='perf_benchmark',
            name=name,
            # track run metadata
            config={
                'model': args.model,
                'time': current_time
            },
        )
        os.environ['WANDB_SILENT'] = 'true'

    while True:
        try:
            benchmark_data: BenchmarkData = benchmark_data_queue.get_nowait()
            benchmark_data_queue.task_done()
            n_benchmark_result += 1
        except asyncio.QueueEmpty:
            if _data_process_completed:
                break
            await asyncio.sleep(1)
            continue
        if start_time is None:
            start_time = benchmark_data.start_time
        total_time = time.perf_counter() - start_time

        if benchmark_data.success:
            n_succeed_queries += 1
            n_query_trunks = len(benchmark_data.chunk_times)
            query_latency = (
                benchmark_data.completed_time - benchmark_data.start_time)
            if n_query_trunks > 1:
                query_first_chunk_latency, query_n_chunks, query_n_chunks_time = (
                    benchmark_data.calculate_query_stream_metric()
                )  # FIXME: Hang when error occurs
            else:
                query_first_chunk_latency = query_latency
                query_n_chunks = 1
                query_n_chunks_time = query_latency

            n_query_prompt_tokens, n_query_completion_tokens = (
                api_plugin.parse_responses(
                    benchmark_data.response_messages,
                    request=benchmark_data.request,
                ))
            n_total_prompt_tokens += n_query_prompt_tokens
            n_total_completion_tokens += n_query_completion_tokens

            total_first_chunk_latency += query_first_chunk_latency
            total_latency += query_latency
            n_total_chunks += query_n_chunks
            total_chunks_time += query_n_chunks_time

            avg_first_chunk_latency = total_first_chunk_latency / n_succeed_queries
            avg_latency = total_latency / n_succeed_queries
            if n_query_trunks > 1:
                n_avg_chunks = (n_total_chunks / n_succeed_queries + 2)
            else:
                n_avg_chunks = n_total_chunks / n_succeed_queries
            avg_chunk_time = total_chunks_time / n_total_chunks
            avg_prompt_tokens = n_total_prompt_tokens / n_succeed_queries
            avg_completion_tokens = n_total_completion_tokens / n_succeed_queries
            avg_token_per_seconds = n_total_completion_tokens / total_time
            avg_time_per_token = total_time / n_total_completion_tokens

            insert_sql = (
                "INSERT INTO result VALUES('%s', %s, '%s', '%s', '%s', %s, %s, %s, %s, %s, %s, %s)"
                % (
                    base64.b64encode(pickle.dumps(
                        benchmark_data.request)).decode('utf-8'),
                    benchmark_data.start_time,
                    json.dumps(benchmark_data.chunk_times, ensure_ascii=False),
                    benchmark_data.success,
                    base64.b64encode(
                        pickle.dumps(
                            benchmark_data.response_messages)).decode('utf-8'),
                    benchmark_data.completed_time,
                    query_latency,
                    query_first_chunk_latency,
                    query_n_chunks,
                    query_n_chunks_time,
                    n_query_prompt_tokens,
                    n_query_completion_tokens,
                ))
        else:
            n_failed_queries += 1
            insert_sql = (
                "INSERT INTO result(request, start_time, chunk_times, success, response_messages, completed_time)\
                VALUES('%s', %s, '%s', '%s', '%s', %s)" % (
                    base64.b64encode(pickle.dumps(
                        benchmark_data.request)).decode('utf-8'),
                    benchmark_data.start_time,
                    json.dumps(benchmark_data.chunk_times, ensure_ascii=False),
                    benchmark_data.success,
                    base64.b64encode(
                        pickle.dumps(
                            benchmark_data.response_messages)).decode('utf-8'),
                    benchmark_data.completed_time,
                ))

        n_total_queries = float(n_succeed_queries + n_failed_queries)
        qps = n_succeed_queries / total_time
        db_cur.execute(insert_sql)
        con.commit()

        default_ndigits = 3
        message = {
            'Time':
            round(total_time, default_ndigits),
            'concurrency':
            concurrency,
            'completed':
            int(n_total_queries),
            'succeed':
            n_succeed_queries,
            'failed':
            n_failed_queries,
            'qps':
            round(qps, default_ndigits),
            'latency':
            round(avg_latency, default_ndigits),
            'time to first token':
            round(avg_first_chunk_latency, default_ndigits),
            'throughput(output tokens per second)':
            round(avg_token_per_seconds, default_ndigits),
            'time per output token':
            round(avg_time_per_token, 5),
            'package per request':
            round(n_avg_chunks, default_ndigits),
            'time per package':
            round(avg_chunk_time, default_ndigits),
            'input tokens per request':
            round(avg_prompt_tokens, default_ndigits),
            'output tokens per request':
            round(avg_completion_tokens, default_ndigits),
        }
        if args.wandb_api_key is not None:
            wandb.log(message)
        if int(n_total_queries) % args.log_every_n_query == 0:
            msg = json.dumps(message, ensure_ascii=False)
            msg = msg[1:-1].replace('"', '')
            logger.info(msg)

    con.commit()
    con.close()
    return (
        total_time,
        n_total_queries,
        n_succeed_queries,
        n_failed_queries,
        qps,
        concurrency,
        avg_latency,
        avg_first_chunk_latency,
        n_avg_chunks,
        avg_chunk_time,
        avg_prompt_tokens,
        avg_completion_tokens,
        avg_token_per_seconds,
        avg_time_per_token,
        result_db_path,
    )


async def benchmark(args) -> None:
    if platform.system() != 'Windows':
        loop = asyncio.get_running_loop()
        for signal_name in {'SIGINT', 'SIGTERM'}:
            loop.add_signal_handler(
                getattr(signal, signal_name),
                functools.partial(signal_handler, signal_name, loop),
            )

    request_tasks: List[asyncio.Task] = []
    request_queue = asyncio.Queue()
    benchmark_data_queue = asyncio.Queue()

    dispatch_task = asyncio.create_task(
        dispatch_requests_worker(request_queue, args))
    statistic_benchmark_metric_task = asyncio.create_task(
        statistic_benchmark_metric_worker(benchmark_data_queue, args))
    for idx, task in enumerate(range(args.parallel)):
        task = asyncio.create_task(
            send_requests_worker(idx, request_queue, benchmark_data_queue,
                                 args))
        request_tasks.append(task)

    expected_number_of_queries = await dispatch_task
    await request_queue.join()
    global _query_send_completed
    _query_send_completed = True
    await asyncio.gather(*request_tasks, return_exceptions=True)
    await benchmark_data_queue.join()
    global _data_process_completed
    _data_process_completed = True
    (
        total_time,
        n_total_queries,
        n_succeed_queries,
        n_failed_queries,
        qps,
        concurrency,
        avg_latency,
        avg_first_chunk_latency,
        n_avg_chunks,
        avg_chunk_time,
        avg_prompt_tokens,
        avg_completion_tokens,
        avg_token_per_seconds,
        avg_time_per_token,
        result_db_path,
    ) = await statistic_benchmark_metric_task

    summary_result(
        expected_number_of_queries,
        total_time,
        n_total_queries,
        n_succeed_queries,
        n_failed_queries,
        qps,
        concurrency,
        avg_latency,
        avg_first_chunk_latency,
        n_avg_chunks,
        avg_chunk_time,
        avg_prompt_tokens,
        avg_completion_tokens,
        avg_token_per_seconds,
        avg_time_per_token,
        result_db_path,
    )
    await asyncio.sleep(0.250)
