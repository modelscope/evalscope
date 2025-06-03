import asyncio
import json
import numpy as np
import platform
import sqlite3
import time
from http import HTTPStatus
from tqdm import tqdm
from typing import AsyncGenerator, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.http_client import AioHttpClient, test_connection
from evalscope.perf.plugin.registry import ApiRegistry, DatasetRegistry
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics
from evalscope.perf.utils.db_util import create_result_table, get_result_db_path, insert_benchmark_data, summary_result
from evalscope.perf.utils.handler import add_signal_handlers, exception_handler
from evalscope.utils.logger import get_logger

logger = get_logger()

data_process_completed_event = asyncio.Event()


@exception_handler
async def get_requests(args: Arguments) -> AsyncGenerator[dict, None]:
    query_generator_class = ApiRegistry(args.api)
    query_generator = query_generator_class(args.tokenizer_path)

    def load_prompt(prompt_path_or_text):
        if prompt_path_or_text.startswith('@'):
            with open(prompt_path_or_text[1:], 'r', encoding='utf-8') as file:
                return file.read()
        return prompt_path_or_text

    async def generate_requests_from_prompt(messages):
        request = query_generator.build_request(messages, args)
        for _ in range(args.number):
            yield request

    async def generate_requests_from_dataset():
        message_generator_class = DatasetRegistry(args.dataset)
        message_generator = message_generator_class(args)

        count = 0
        for messages in message_generator:
            request = query_generator.build_request(messages, args)
            if request is not None:
                yield request
                count += 1
                if args.number and count >= args.number:
                    break

    if args.prompt:
        prompt = load_prompt(args.prompt)
        messages = [{'role': 'user', 'content': prompt}] if args.apply_chat_template else prompt
        generator = generate_requests_from_prompt(messages)
    elif args.dataset:
        generator = generate_requests_from_dataset()
    else:
        raise Exception('Either prompt or dataset is required!')

    async for request in generator:
        yield request
        if args.rate != -1:
            interval = np.random.exponential(1.0 / args.rate)
            await asyncio.sleep(interval)


@exception_handler
async def send_request(
    semaphore: asyncio.Semaphore,
    request: dict,
    benchmark_data_queue: asyncio.Queue,
    args: Arguments,
):
    async with semaphore:
        client = AioHttpClient(args)
        async with client:
            benchmark_data = BenchmarkData(request=request)
            benchmark_data.start_time = time.perf_counter()
            collected_messages = []
            try:
                async for is_error, state_code, response_data in client.post(request):
                    if is_error or state_code != HTTPStatus.OK:
                        logger.error(f'Request: {request} failed, state_code: {state_code}, data: {response_data}')
                        benchmark_data.success = False
                        break
                    if response_data:
                        collected_messages.append(response_data)
                        benchmark_data.chunk_times.append(time.perf_counter())
                        benchmark_data.success = True
                        benchmark_data.update_gpu_usage()
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
async def statistic_benchmark_metric(benchmark_data_queue: asyncio.Queue, args: Arguments):
    metrics = BenchmarkMetrics(concurrency=args.parallel)

    api_plugin_class = ApiRegistry(args.api)
    api_plugin = api_plugin_class(args.tokenizer_path)

    result_db_path = get_result_db_path(args)

    collected_benchmark_data = []

    with tqdm(desc='Processing', total=args.number) as pbar:
        while not (data_process_completed_event.is_set() and benchmark_data_queue.empty()):
            try:
                # Attempt to get benchmark data from the queue with a timeout
                benchmark_data = await asyncio.wait_for(benchmark_data_queue.get(), timeout=0.01)
                benchmark_data_queue.task_done()
            except asyncio.TimeoutError:
                # If timeout, continue to the next iteration
                continue

            # Update metrics based on the benchmark data
            metrics.update_metrics(benchmark_data, api_plugin)

            # Collect benchmark data for later database insertion
            collected_benchmark_data.append(benchmark_data)

            # Create a message with the updated metrics
            message = metrics.create_message()

            # Log the message to wandb\swanlab if the api key is provided
            if args.wandb_api_key:
                import wandb
                wandb.log(message)
            if args.swanlab_api_key:
                import swanlab
                swanlab.log(message)

            # Log the message to the logger every n queries
            if int(metrics.n_total_queries) % args.log_every_n_query == 0:
                msg = json.dumps(message, ensure_ascii=False, indent=2)
                logger.info(msg)

            pbar.update(1)  # Update the progress bar

    # Now perform database operations after all benchmark data has been processed
    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        create_result_table(cursor)
        for benchmark_data in collected_benchmark_data:
            insert_benchmark_data(cursor, benchmark_data)
        con.commit()

    return metrics, result_db_path


@exception_handler
async def connect_test(args: Arguments) -> bool:
    if (not args.no_test_connection) and (not await test_connection(args)):
        raise TimeoutError('Test connection failed')


@exception_handler
async def benchmark(args: Arguments) -> Tuple[Dict, Dict]:
    if platform.system() != 'Windows':
        loop = asyncio.get_running_loop()
        add_signal_handlers(loop)

    # init queue
    benchmark_data_queue = asyncio.Queue()
    # reset event
    data_process_completed_event.clear()
    # test connection
    await connect_test(args)
    # start statistic benchmark metric
    statistic_benchmark_metric_task = asyncio.create_task(statistic_benchmark_metric(benchmark_data_queue, args))
    # start send request
    semaphore = asyncio.Semaphore(args.parallel)
    send_request_tasks: List[asyncio.Task] = []
    async for request in get_requests(args):
        task = asyncio.create_task(send_request(semaphore, request, benchmark_data_queue, args))
        send_request_tasks.append(task)

    await asyncio.gather(*send_request_tasks, return_exceptions=True)
    await benchmark_data_queue.join()
    data_process_completed_event.set()

    metrics, result_db_path = await statistic_benchmark_metric_task
    metrics_result, percentile_result = summary_result(args, metrics, result_db_path)
    return metrics_result, percentile_result
