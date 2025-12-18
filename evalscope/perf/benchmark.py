import asyncio
import json
import numpy as np
import sqlite3
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Tuple

from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm
from .arguments import Arguments
from .http_client import AioHttpClient, test_connection
from .plugin import ApiRegistry, DatasetRegistry
from .utils.benchmark_util import BenchmarkMetrics
from .utils.db_util import create_result_table, get_result_db_path, insert_benchmark_data, load_prompt, summary_result
from .utils.handler import exception_handler
from .utils.log_utils import maybe_log_to_visualizer

if TYPE_CHECKING:
    from .plugin import ApiPluginBase

logger = get_logger()

data_process_completed_event = asyncio.Event()


@exception_handler
async def get_requests(args: Arguments, api_plugin: 'ApiPluginBase') -> AsyncGenerator[dict, None]:

    async def generate_requests_from_prompt():
        prompt = load_prompt(args.prompt)
        messages = [{'role': 'user', 'content': prompt}] if args.apply_chat_template else prompt
        request = api_plugin.build_request(messages)
        for _ in range(args.number):
            yield request

    async def generate_requests_from_dataset():
        message_generator_class = DatasetRegistry.get_class(args.dataset)
        message_generator = message_generator_class(args)

        dataset_messages = []
        try:
            for messages in message_generator.build_messages():
                dataset_messages.append(messages)
                if len(dataset_messages) >= args.number:
                    break
        except StopIteration:
            pass

        if not dataset_messages:
            raise Exception('Dataset is empty!')

        count = 0
        dataset_index = 0

        while count < args.number:
            messages = dataset_messages[dataset_index]
            request = api_plugin.build_request(messages)
            if request is not None:
                yield request
                count += 1

            dataset_index = (dataset_index + 1) % len(dataset_messages)

    if args.prompt:
        generator = generate_requests_from_prompt()
    elif args.dataset:
        generator = generate_requests_from_dataset()
    else:
        raise ValueError('Either prompt or dataset is required!')

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
    client: AioHttpClient,  # reuse shared client
):
    async with semaphore:
        benchmark_data = await client.post(request)
        benchmark_data.update_gpu_usage()
        await benchmark_data_queue.put(benchmark_data)


@exception_handler
async def statistic_benchmark_metric(benchmark_data_queue: asyncio.Queue, args: Arguments, api_plugin: 'ApiPluginBase'):
    metrics = BenchmarkMetrics(concurrency=args.parallel, rate=args.rate)
    result_db_path = get_result_db_path(args)

    # Stream inserts to DB to avoid accumulating all results in memory
    commit_every = args.db_commit_interval
    processed_since_commit = 0

    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        create_result_table(cursor)

        with tqdm(desc='Processing', total=args.number, logger=logger, log_interval=HEARTBEAT_INTERVAL_SEC) as pbar:
            while not (data_process_completed_event.is_set() and benchmark_data_queue.empty()):
                try:
                    benchmark_data = await asyncio.wait_for(benchmark_data_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # Update metrics and write to DB immediately
                metrics.update_metrics(benchmark_data, api_plugin)
                insert_benchmark_data(cursor, benchmark_data)
                processed_since_commit += 1
                if processed_since_commit >= commit_every:
                    con.commit()
                    processed_since_commit = 0

                message = metrics.create_message()

                maybe_log_to_visualizer(args, message)

                if int(metrics.n_total_queries) % args.log_every_n_query == 0:
                    msg = json.dumps(message, ensure_ascii=False, indent=2)
                    logger.info(msg)

                benchmark_data_queue.task_done()
                pbar.update(1)

        con.commit()

    return metrics, result_db_path


@exception_handler
async def connect_test(args: Arguments, api_plugin) -> bool:
    if (not args.no_test_connection) and (not await test_connection(args, api_plugin)):
        raise TimeoutError('Test connection failed')


@exception_handler
async def benchmark(args: Arguments) -> Tuple[Dict, Dict]:
    api_plugin_class = ApiRegistry.get_class(args.api)
    api_plugin = api_plugin_class(args)

    benchmark_data_queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, args.parallel * args.queue_size_multiplier))
    data_process_completed_event.clear()

    # test connection
    await connect_test(args, api_plugin)

    # Create a single shared client session for all requests
    client = AioHttpClient(args, api_plugin)
    async with client:
        # start statistic benchmark metric (consumer)
        statistic_benchmark_metric_task = asyncio.create_task(
            statistic_benchmark_metric(benchmark_data_queue, args, api_plugin)
        )

        # start sending requests with bounded in-flight tasks
        semaphore = asyncio.Semaphore(args.parallel)
        in_flight: set[asyncio.Task] = set()
        max_in_flight = args.parallel * args.in_flight_task_multiplier

        async for request in get_requests(args, api_plugin):
            # Keep the number of scheduled tasks bounded to avoid OOM
            if len(in_flight) >= max_in_flight:
                done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                in_flight = pending

            task = asyncio.create_task(send_request(semaphore, request, benchmark_data_queue, args, client))
            in_flight.add(task)

        # Wait for remaining in-flight tasks
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

        # Drain queue and finish
        await benchmark_data_queue.join()
        data_process_completed_event.set()

        metrics, result_db_path = await statistic_benchmark_metric_task

    metrics_result, percentile_result = summary_result(args, metrics, result_db_path)
    return metrics_result, percentile_result
