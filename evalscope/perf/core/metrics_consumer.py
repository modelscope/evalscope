import asyncio
import json
import sqlite3
from typing import TYPE_CHECKING, Tuple

from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient, test_connection
from evalscope.perf.utils.benchmark_util import Metrics, MetricsAccumulator
from evalscope.perf.utils.db_util import create_result_table, get_result_db_path, insert_benchmark_data
from evalscope.perf.utils.handler import exception_handler
from evalscope.perf.utils.log_utils import maybe_log_to_visualizer
from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

if TYPE_CHECKING:
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()

# Global event signalling that all requests have been dispatched and the
# metrics consumer should flush remaining items and exit.
data_process_completed_event = asyncio.Event()


@exception_handler
async def statistic_benchmark_metric(
    benchmark_data_queue: asyncio.Queue,
    args: Arguments,
    api_plugin: 'ApiPluginBase',
) -> Tuple['MetricsAccumulator', str]:
    """Consume benchmark results from the queue, update metrics, and persist to DB.

    Args:
        benchmark_data_queue: Queue populated by request workers.
        args: Benchmark configuration.
        api_plugin: API plugin used to finalise token counts.

    Returns:
        Tuple of ``(metrics_accumulator_result, result_db_path)``.
    """
    accumulator = MetricsAccumulator(concurrency=args.parallel, rate=args.rate)
    result_db_path = get_result_db_path(args)

    # Stream inserts to DB to avoid accumulating all results in memory.
    commit_every = args.db_commit_interval
    processed_since_commit = 0

    with sqlite3.connect(result_db_path, check_same_thread=False) as con:
        cursor = con.cursor()
        create_result_table(cursor)

        cur_run_name = (
            f'rate_{args.rate}_number_{args.number}'
            if args.open_loop else f'parallel_{args.parallel}_number_{args.number}'
        )
        with tqdm(
            desc=f'Processing[{cur_run_name}]',
            total=args.number,
            logger=logger,
            log_interval=HEARTBEAT_INTERVAL_SEC,
            track_progress=True,
        ) as pbar:
            while not (data_process_completed_event.is_set() and benchmark_data_queue.empty()):
                try:
                    benchmark_data = await asyncio.wait_for(benchmark_data_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # Update accumulator and write to DB immediately.
                accumulator.update(benchmark_data, api_plugin)
                insert_benchmark_data(cursor, benchmark_data)
                processed_since_commit += 1
                if processed_since_commit >= commit_every:
                    await asyncio.to_thread(con.commit)
                    processed_since_commit = 0

                message = accumulator.to_result().create_message(api_type=args.api)

                await asyncio.to_thread(maybe_log_to_visualizer, args, message)

                if int(accumulator.n_total) % args.log_every_n_query == 0:
                    msg = json.dumps(message, ensure_ascii=False, indent=2)
                    logger.info(msg)

                benchmark_data_queue.task_done()
                pbar.update(1)

        await asyncio.to_thread(con.commit)

    return accumulator.to_result(), result_db_path


@exception_handler
async def connect_test(args: Arguments, api_plugin: 'ApiPluginBase') -> None:
    """Perform a connection test unless disabled or not applicable.

    Raises:
        TimeoutError: If the test connection fails.
    """
    if Metrics.is_embedding_or_rerank(args.api):
        return

    if args.no_test_connection:
        return

    if not await test_connection(args, api_plugin):
        raise TimeoutError('Test connection failed')
