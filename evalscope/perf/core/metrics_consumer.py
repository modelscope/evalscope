import asyncio
import json
import sqlite3
from tqdm import tqdm as tqdm_std
from typing import TYPE_CHECKING, Tuple

from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient, test_connection
from evalscope.perf.utils.benchmark_util import Metrics, MetricsAccumulator
from evalscope.perf.utils.db_util import create_result_table, get_result_db_path, insert_benchmark_data
from evalscope.perf.utils.handler import exception_handler
from evalscope.perf.utils.log_utils import maybe_log_to_visualizer
from evalscope.perf.utils.trace_metrics import TraceAccumulator, TraceLevelSummary
from evalscope.perf.utils.workload_timeline import WorkloadTimeline
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
) -> Tuple['MetricsAccumulator', 'TraceLevelSummary', 'WorkloadTimeline', str]:
    """Consume benchmark results from the queue, update metrics, and persist to DB.

    Args:
        benchmark_data_queue: Queue populated by request workers.
        args: Benchmark configuration.
        api_plugin: API plugin used to finalise token counts.

    Returns:
        Tuple of ``(metrics_accumulator_result, trace_level_summary, workload_timeline, result_db_path)``.
        ``trace_level_summary`` is empty for single-turn runs (no ``trace_id``);
        ``workload_timeline`` always accumulates regardless of mode, callers
        may inspect ``n_points`` before rendering downstream tables.
    """
    accumulator = MetricsAccumulator(concurrency=args.parallel, rate=args.rate)
    trace_acc = TraceAccumulator()
    workload_timeline = WorkloadTimeline()
    result_db_path = get_result_db_path(args)
    warmup_count = args.warmup_count

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

        # Warmup bar
        _warmup_pbar = None
        if warmup_count > 0:
            _warmup_pbar = tqdm_std(
                desc=f'Warmup[{cur_run_name}]',
                total=warmup_count,
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

                if benchmark_data.is_warmup:
                    benchmark_data_queue.task_done()
                    # Multi-turn: only count last turn per conversation.
                    if not benchmark_data.is_last_turn and benchmark_data.input_num_turns > 0:
                        continue
                    if _warmup_pbar:
                        _warmup_pbar.update(1)
                    continue

                # First benchmark item — close the warmup bar so it disappears.
                if _warmup_pbar:
                    _warmup_pbar.close()
                    _warmup_pbar = None

                # Update accumulator and write to DB immediately.
                accumulator.update(benchmark_data, api_plugin)
                # Feed the per-trace accumulator *after* MetricsAccumulator.update
                # so finalize() has populated prompt/completion tokens (idempotent).
                # Single-turn items (trace_id is None) are silently skipped inside.
                trace_acc.feed(benchmark_data)
                # Workload timeline tracks cumulative tokens vs time for the
                # Overall / Last-window / Steady-state throughput breakdown.
                workload_timeline.feed(benchmark_data)
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
                # In multi-turn mode each conversation produces multiple turns;
                # advance the progress bar only once per conversation (on the
                # last turn).  In single-turn mode is_last_turn is always False,
                # so we fall back to updating on every item.
                if not benchmark_data.is_last_turn and benchmark_data.input_num_turns > 0:
                    continue
                pbar.update(1)

        await asyncio.to_thread(con.commit)

    return accumulator.to_result(), trace_acc.to_summary(), workload_timeline, result_db_path


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
