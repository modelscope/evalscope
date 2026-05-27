"""Multi-turn conversation benchmark for evalscope perf.

Execution model
---------------
``parallel`` asyncio tasks (workers) run concurrently.  Each worker owns one
active conversation at a time and progresses through its turns sequentially:

::

    Conversation Pool: [conv_0, conv_1, ..., conv_M]  (cycled until turn budget)
             |
       +-----+------+
     Worker_0    Worker_1  ...  Worker_(parallel-1)
      |                |
     conv_i turn_0   conv_j turn_0
      | (await)       | (await)          <- parallel turns in-flight simultaneously
     conv_i turn_1   conv_j turn_1
      |               |
     ...             ...

Parameter semantics (same as normal benchmark mode)
----------------------------------------------------
* ``--number``   - total number of conversations to run.
* ``--parallel`` - number of conversations executed concurrently.

Multi-turn specific parameters
-------------------------------
* ``--min-turns`` - minimum user turns per conversation (random_multi_turn).
* ``--max-turns`` - maximum user turns per conversation.

Note on open-loop mode
----------------------
Open-loop mode is **not** supported for multi-turn benchmarks.  The core issue
is that open-loop semantics require each request to be dispatched independently
of whether previous requests have completed.  Multi-turn conversations, however,
have a hard sequential dependency: turn N must wait for the assistant response
from turn N-1 before the next request can be constructed (the response is
appended to the context).  Removing this dependency would break the conversation
context and produce meaningless evaluation results.
"""

import asyncio
from typing import TYPE_CHECKING, List, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.core.metrics_consumer import connect_test, data_process_completed_event, statistic_benchmark_metric
from evalscope.perf.core.strategies import MultiTurnStrategy
from evalscope.perf.plugin import ApiRegistry, DatasetRegistry
from evalscope.perf.plugin.datasets.base import Conversation
from evalscope.perf.utils.db_util import summary_result
from evalscope.perf.utils.handler import exception_handler

if TYPE_CHECKING:
    from evalscope.perf.utils.perf_models import BenchmarkSummary, PercentileResult
    from evalscope.perf.utils.trace_metrics import TraceLevelSummary
    from evalscope.perf.utils.workload_timeline import WorkloadThroughput

from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

logger = get_logger()


@exception_handler
async def run_multi_turn_benchmark(
    args: Arguments,
) -> Tuple['BenchmarkSummary', 'PercentileResult', Optional['TraceLevelSummary'], Optional['WorkloadThroughput']]:
    """Run a multi-turn conversation benchmark.

    Args:
        args: Benchmark configuration.  ``args.number`` is the total number of
              conversations to run and ``args.parallel`` controls concurrency
              (number of simultaneously active conversations).

    Returns:
        4-tuple of ``(summary, percentiles, trace_summary, workload_throughput)``.
    """
    api_plugin_class = ApiRegistry.get_class(args.api)
    api_plugin = api_plugin_class(args)

    # ------------------------------------------------------------------
    # 1. Load all conversations from the dataset
    # ------------------------------------------------------------------
    logger.info(f'Loading conversations from dataset: {args.dataset}')
    dataset_plugin = DatasetRegistry.get_class(args.dataset)(args)

    # Cap preloading to the total conversation count (benchmark + warmup)
    # so the strategy has enough conversations without cycling warmup prompts
    # into the benchmark portion.
    _max_preload = args.total_count
    with tqdm(desc='Loading[conversations]', logger=logger) as pbar:
        all_conversations: List[Conversation] = []
        for conv in dataset_plugin.build_messages():
            all_conversations.append(conv)
            pbar.update(1)
            if len(all_conversations) >= _max_preload:
                break

    if not all_conversations:
        raise ValueError(f'Dataset "{args.dataset}" produced no conversations!')

    logger.info(f'Loaded {len(all_conversations)} conversations')

    # ------------------------------------------------------------------
    # 2. Setup shared state
    # ------------------------------------------------------------------
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, args.parallel * args.queue_size_multiplier))
    data_process_completed_event.clear()

    # ------------------------------------------------------------------
    # 3. Test connection
    # ------------------------------------------------------------------
    await connect_test(args, api_plugin)

    # ------------------------------------------------------------------
    # 4. Shared HTTP client
    # ------------------------------------------------------------------
    client = AioHttpClient(args, api_plugin)

    async with client:
        # ----------------------------------------------------------------
        # 5. Start the metrics consumer task
        # ----------------------------------------------------------------
        statistic_task = asyncio.create_task(statistic_benchmark_metric(queue, args, api_plugin))

        # ----------------------------------------------------------------
        # 6. Run multi-turn strategy
        # ----------------------------------------------------------------
        strategy = MultiTurnStrategy(
            args=args,
            api_plugin=api_plugin,
            client=client,
            queue=queue,
            all_conversations=all_conversations,
        )
        await strategy.run()

        # ----------------------------------------------------------------
        # 7. Drain the metrics queue and signal the consumer to stop
        # ----------------------------------------------------------------
        await queue.join()
        data_process_completed_event.set()

        metrics, trace_summary, workload_timeline, result_db_path = await statistic_task

    # ------------------------------------------------------------------
    # 8. Summarise and return
    # ------------------------------------------------------------------
    return summary_result(
        args,
        metrics,
        result_db_path,
        trace_summary=trace_summary,
        workload_timeline=workload_timeline,
    )
