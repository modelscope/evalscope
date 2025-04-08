import asyncio
import copy
import json
import numpy as np
import os
import platform
import sqlite3
import threading
import time
from http import HTTPStatus
from tqdm import tqdm
from typing import AsyncGenerator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.http_client import AioHttpClient, test_connection
from evalscope.perf.plugin.registry import ApiRegistry, DatasetRegistry
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics
from evalscope.perf.utils.db_util import create_result_table, get_result_db_path, insert_benchmark_data, summary_result
from evalscope.perf.utils.handler import add_signal_handlers, exception_handler
from evalscope.perf.utils.local_server import start_app
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
        messages = [{'role': 'user', 'content': prompt}]
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
async def statistic_benchmark_metric_worker(benchmark_data_queue: asyncio.Queue, args: Arguments):
    metrics = BenchmarkMetrics(concurrency=args.parallel)

    api_plugin_class = ApiRegistry(args.api)
    api_plugin = api_plugin_class(args.tokenizer_path)

    result_db_path = get_result_db_path(args)
    # Initialize wandb if the api key is provided
    if args.wandb_api_key:
        import datetime
        try:
            import wandb
        except ImportError:
            raise RuntimeError('Cannot import wandb. Please install it with command: \n pip install wandb')
        os.environ['WANDB_SILENT'] = 'true'
        os.environ['WANDB_DIR'] = args.outputs_dir

        wandb.login(key=args.wandb_api_key)
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = args.name if args.name else f'{args.model_id}_{current_time}'
        wandb.init(project='perf_benchmark', name=name, config=args.to_dict())
    # Initialize SwanLab if the api key is provided
    if args.swanlab_api_key:
        import datetime
        try:
            import swanlab
        except ImportError:
            raise RuntimeError('Cannot import swanlab. Please install it with command: \n pip install swanlab')
        os.environ['SWANLAB_LOG_DIR'] = args.outputs_dir
        if not args.swanlab_api_key == 'local':
            swanlab.login(api_key=args.swanlab_api_key)
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = args.name if args.name else f'{args.model_id}_{current_time}'
        swanlab.config.update({'framework': '📏evalscope'})
        swanlab.init(
            project='perf_benchmark',
            name=name,
            config=args.to_dict(),
            mode='local' if args.swanlab_api_key == 'local' else None)

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
                wandb.log(message)
            if args.swanlab_api_key:
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
async def start_server(args: Arguments) -> bool:
    if args.api.startswith('local'):
        #  start local server
        server = threading.Thread(target=start_app, args=(copy.deepcopy(args), ), daemon=True)
        server.start()

        if args.dataset.startswith('speed_benchmark'):
            args.url = f'http://127.0.0.1:{args.port}/v1/completions'
        else:
            args.url = f'http://127.0.0.1:{args.port}/v1/chat/completions'

    if (not args.no_test_connection) and (not await test_connection(args)):
        raise TimeoutError('Test connection failed')


@exception_handler
async def benchmark(args: Arguments) -> None:
    if platform.system() != 'Windows':
        loop = asyncio.get_running_loop()
        add_signal_handlers(loop)

    # init queue
    benchmark_data_queue = asyncio.Queue()

    # reset event
    data_process_completed_event.clear()

    semaphore = asyncio.Semaphore(args.parallel)

    async def create_send_request_tasks():
        tasks: List[asyncio.Task] = []
        async for request in get_requests(args):
            task = asyncio.create_task(send_request(semaphore, request, benchmark_data_queue, args))
            tasks.append(task)
        return tasks

    async def run_tasks():
        await start_server(args)

        statistic_benchmark_metric_task = asyncio.create_task(
            statistic_benchmark_metric_worker(benchmark_data_queue, args))
        send_request_tasks = await create_send_request_tasks()

        await asyncio.gather(*send_request_tasks, return_exceptions=True)
        await benchmark_data_queue.join()
        data_process_completed_event.set()

        metrics, result_db_path = await statistic_benchmark_metric_task
        summary_result(args, metrics, result_db_path)

    await run_tasks()
