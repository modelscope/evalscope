import asyncio
import os
import platform
import sqlite3
import sys
import time
from http import HTTPStatus
from typing import List

import json
import numpy as np

from evalscope.perf.arguments import Arguments
from evalscope.perf.http_client import AioHttpClient
from evalscope.perf.plugin.registry import api_registry, dataset_registry
from evalscope.perf.utils._logging import logger
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics
from evalscope.perf.utils.db_utils import create_result_table, get_result_db_path, insert_benchmark_data, summary_result
from evalscope.perf.utils.exception_handler import exception_handler
from evalscope.perf.utils.signal_handler import add_signal_handlers

query_send_completed_event = asyncio.Event()
data_process_completed_event = asyncio.Event()


@exception_handler
async def dispatch_requests_worker(request_queue: asyncio.Queue, args: Arguments):
    query_generator_class = api_registry(args.api)
    if not query_generator_class:
        logger.info('Can not find query generator: %s' % args.api)
    query_generator = query_generator_class(args.tokenizer_path)
    total_query_counter = 0
    if args.prompt is not None:
        if args.prompt.startswith('@'):
            with open(args.prompt, 'r', encoding='utf-8') as f:
                prompt = f.read()
        else:
            prompt = args.prompt
        messages = [{'role': 'user', 'content': prompt}]
        request = query_generator.build_request(messages, args)
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
                logger.info('Can not find dataset: %s plugin.' % (args.dataset))
                sys.exit(1)
            message_generator = message_generator_class(args)
            for messages in message_generator:
                request = query_generator.build_request(messages, args)
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


@exception_handler
async def send_requests_worker(
    task_id,
    request_queue: asyncio.Queue,
    benchmark_data_queue: asyncio.Queue,
    args: Arguments,
):
    client = AioHttpClient(
        args.url,
        conn_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        headers=args.headers,
        debug=args.debug,
    )
    async with client:
        while True:
            if query_send_completed_event.is_set() and request_queue.empty():
                break
            try:
                request = await asyncio.wait_for(request_queue.get(), timeout=0.1)
                request_queue.task_done()
            except asyncio.TimeoutError:
                continue

            # auto get start_time when initializing
            benchmark_data = BenchmarkData(request=request)
            collected_messages = []
            try:
                async for is_error, state_code, response_data in client.post(request):
                    if is_error or state_code != HTTPStatus.OK:
                        logger.error(f'Request: {request} failed, state_code: {state_code}, data: {response_data}')
                        benchmark_data.success = False
                        break
                    if response_data:
                        logger.info(response_data)
                        collected_messages.append(response_data)
                        benchmark_data.chunk_times.append(time.perf_counter())
                    benchmark_data.success = True
            except Exception as e:
                if response_data:
                    collected_messages.append(response_data)
                benchmark_data.success = False
                logger.exception(e)
                logger.error(f'Request query: {request} exception')
            finally:
                benchmark_data.completed_time = time.perf_counter()
                benchmark_data.response_messages = collected_messages
                await benchmark_data_queue.put(benchmark_data)


@exception_handler
async def statistic_benchmark_metric_worker(benchmark_data_queue: asyncio.Queue, args: Arguments):
    metrics = BenchmarkMetrics(concurrency=args.parallel)

    api_plugin_class = api_registry(args.api)
    if not api_plugin_class:
        logger.error('Can not find query generator: %s' % args.api)
    api_plugin = api_plugin_class(args.tokenizer_path)

    result_db_path = get_result_db_path(args.name, args.model)
    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        create_result_table(cursor)

        if args.wandb_api_key:
            import wandb
            import datetime
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            name = args.name if args.name else f'{args.model}_{current_time}'
            wandb.init(project='perf_benchmark', name=name, config={'model': args.model, 'time': current_time})
            os.environ['WANDB_SILENT'] = 'true'

        while True:
            if data_process_completed_event.is_set() and benchmark_data_queue.empty():
                break
            try:
                benchmark_data = await asyncio.wait_for(benchmark_data_queue.get(), timeout=0.01)
                benchmark_data_queue.task_done()
            except asyncio.TimeoutError:
                continue

            metrics.update_metrics(benchmark_data, api_plugin)
            insert_benchmark_data(cursor, benchmark_data)
            con.commit()

            message = metrics.create_message()
            if args.wandb_api_key:
                wandb.log(message)
            if int(metrics.n_total_queries) % args.log_every_n_query == 0:
                msg = json.dumps(message, ensure_ascii=False)
                msg = msg[1:-1].replace('"', '')
                logger.info(msg)

    return metrics, result_db_path


@exception_handler
async def benchmark(args: Arguments) -> None:
    if platform.system() != 'Windows':
        loop = asyncio.get_running_loop()
        add_signal_handlers(loop)

    request_queue = asyncio.Queue()
    benchmark_data_queue = asyncio.Queue()

    async def create_send_request_tasks():
        tasks: List[asyncio.Task] = []
        for idx in range(args.parallel):
            task = asyncio.create_task(send_requests_worker(idx, request_queue, benchmark_data_queue, args))
            tasks.append(task)
        return tasks

    async def run_tasks():
        dispatch_task = asyncio.create_task(dispatch_requests_worker(request_queue, args))
        statistic_benchmark_metric_task = asyncio.create_task(
            statistic_benchmark_metric_worker(benchmark_data_queue, args))
        send_request_tasks = await create_send_request_tasks()

        expected_number_of_queries = await dispatch_task
        await request_queue.join()
        query_send_completed_event.set()

        await asyncio.gather(*send_request_tasks, return_exceptions=True)
        await benchmark_data_queue.join()
        data_process_completed_event.set()

        metrics, result_db_path = await statistic_benchmark_metric_task
        summary_result(metrics, expected_number_of_queries, result_db_path)

        await asyncio.sleep(0.250)

    await run_tasks()
