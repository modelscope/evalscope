import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from evalscope.perf.arguments import Arguments

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase


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
