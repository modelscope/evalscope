import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Iterable, List, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


class BenchmarkStrategy(ABC):
    """Abstract base class for benchmark execution strategies.

    Subclasses implement the ``run`` coroutine which drives the actual
    request-dispatch loop.  All strategies share the same constructor
    signature so that callers can swap strategies transparently.

    Args:
        args: Benchmark configuration.
        api_plugin: API plugin instance.
        client: Shared HTTP client (already entered as async context manager).
        queue: Queue to which completed :class:`~evalscope.perf.utils.benchmark_util.BenchmarkData`
               objects are pushed for downstream metric collection.
    """

    def __init__(
        self,
        args: Arguments,
        api_plugin: 'ApiPluginBase',
        client: 'AioHttpClient',
        queue: asyncio.Queue,
    ) -> None:
        self.args = args
        self.api_plugin = api_plugin
        self.client = client
        self.queue = queue

    @abstractmethod
    async def run(self) -> None:
        """Dispatch all requests and push results to ``self.queue``.

        The caller is responsible for:
        - awaiting ``queue.join()`` after this coroutine returns.
        - setting ``data_process_completed_event`` to signal the consumer.
        """

    @staticmethod
    def _compute_deadline(duration: Optional[float]) -> Optional[float]:
        """Translate a duration (seconds, may be ``None``) into a ``perf_counter`` deadline."""
        return None if duration is None else time.perf_counter() + duration

    @staticmethod
    async def _gather_with_deadline(
        tasks: Iterable[asyncio.Task],
        deadline: Optional[float] = None,
    ) -> None:
        """Await ``tasks`` with optional hard-cancel at ``deadline``.

        Behaves like ``asyncio.gather(*tasks, return_exceptions=True)`` when
        ``deadline`` is ``None``.  Otherwise, if the deadline is exceeded while
        waiting, all still-pending tasks are cancelled and re-gathered to drain
        their cancellations.  Matches the trie-style "stop new dispatches at
        deadline, drop in-flight" semantic shared by all benchmark strategies.
        """
        tasks = list(tasks)
        if not tasks:
            return
        if deadline is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return
        remaining = max(0.0, deadline - time.perf_counter())
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            logger.info(f'Duration deadline reached; cancelling {len(tasks)} in-flight task(s).')
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _partition_requests(gen: AsyncIterator[Tuple[dict, bool]], ) -> Tuple[List[dict], List[dict]]:
        """Consume all ``(request, is_warmup)`` items from gen.

        Returns:
            (warmup_requests, benchmark_requests) – two plain lists that can
            be iterated independently, enabling a clean two-phase dispatch
            without any buffer/closure complexity.
        """
        warmup: List[dict] = []
        benchmark: List[dict] = []
        async for request, is_warmup in gen:
            if is_warmup:
                warmup.append(request)
            else:
                benchmark.append(request)
        return warmup, benchmark
