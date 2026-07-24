import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.metrics_consumer import statistic_benchmark_metric
from evalscope.utils.asyncio_runtime import cancel_and_wait

if TYPE_CHECKING:
    from evalscope.perf.plugin.api.base import ApiPluginBase
    from evalscope.perf.utils.benchmark_util import MetricsAccumulator
    from evalscope.perf.utils.trace_metrics import TraceLevelSummary
    from evalscope.perf.utils.workload_timeline import WorkloadTimeline


async def run_benchmark_pipeline(
    producer: Coroutine[Any, Any, None],
    benchmark_data_queue: asyncio.Queue,
    args: Arguments,
    api_plugin: 'ApiPluginBase',
) -> Tuple['MetricsAccumulator', 'TraceLevelSummary', 'WorkloadTimeline', str]:
    """Run one producer and its metrics consumer with coordinated cleanup."""
    completed_event = asyncio.Event()
    producer_task = asyncio.create_task(producer)
    consumer_task = asyncio.create_task(
        statistic_benchmark_metric(benchmark_data_queue, args, api_plugin, completed_event)
    )
    drain_task: Optional[asyncio.Task[None]] = None

    try:
        done, _ = await asyncio.wait((producer_task, consumer_task), return_when=asyncio.FIRST_COMPLETED)
        if consumer_task in done:
            await consumer_task
            raise RuntimeError('Metrics consumer exited before request production completed.')

        await producer_task
        drain_task = asyncio.create_task(benchmark_data_queue.join())
        done, _ = await asyncio.wait((drain_task, consumer_task), return_when=asyncio.FIRST_COMPLETED)
        if consumer_task in done:
            await consumer_task
            raise RuntimeError('Metrics consumer exited before the result queue was drained.')

        await drain_task
        completed_event.set()
        return await consumer_task
    finally:
        completed_event.set()
        tasks = [producer_task, consumer_task]
        if drain_task is not None:
            tasks.append(drain_task)
        await asyncio.gather(*(cancel_and_wait(task) for task in tasks), return_exceptions=True)
