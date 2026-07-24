import asyncio
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Protocol, TypeVar

from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


class _AsyncCloseable(Protocol):

    async def close(self) -> None:
        ...


_AsyncClientT = TypeVar('_AsyncClientT', bound=_AsyncCloseable)


@dataclass
class _ClientEntry(Generic[_AsyncClientT]):
    client: _AsyncClientT
    close_task: Optional[asyncio.Task[None]] = None


class LoopBoundAsyncClientPool(Generic[_AsyncClientT]):
    """Own one async client per event loop and close each on its owner loop."""

    def __init__(self, factory: Callable[[], _AsyncClientT]) -> None:
        self._factory = factory
        self._clients: Dict[asyncio.AbstractEventLoop, _ClientEntry[_AsyncClientT]] = {}
        self._lock = threading.Lock()

    def get(self) -> _AsyncClientT:
        """Return the client associated with the running event loop."""
        loop = asyncio.get_running_loop()
        with self._lock:
            entry = self._clients.get(loop)
            if entry is None:
                client = self._factory()
                entry = _ClientEntry(client)
                self._clients[loop] = entry
                AsyncioLoopRunner.register_close_callback(lambda: self._close_on_owner_loop(loop, entry))
            elif entry.close_task is not None:
                raise RuntimeError('The async model client for this event loop is closing.')
            return entry.client

    async def aclose(self) -> None:
        """Close all cached clients on their respective owner loops."""
        with self._lock:
            clients = list(self._clients.items())

        errors: List[Exception] = []
        for loop, entry in clients:
            try:
                await self._close_client(loop, entry)
            except Exception as e:
                errors.append(e)

        if errors:
            raise errors[0]

    async def _close_client(
        self,
        loop: asyncio.AbstractEventLoop,
        entry: _ClientEntry[_AsyncClientT],
    ) -> None:
        running_loop = asyncio.get_running_loop()
        if loop is running_loop:
            await self._close_on_owner_loop(loop, entry)
            return

        if loop.is_closed() or not loop.is_running():
            raise RuntimeError(
                'Cannot close an async model client after its owner event loop has stopped. '
                'Call ModelAPI.aclose() before shutting down the loop.'
            )

        future = asyncio.run_coroutine_threadsafe(self._close_on_owner_loop(loop, entry), loop)
        try:
            await asyncio.wrap_future(future)
        except BaseException:
            if not future.done():
                future.cancel()
            raise

    async def _close_on_owner_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        entry: _ClientEntry[_AsyncClientT],
    ) -> None:
        with self._lock:
            if self._clients.get(loop) is not entry:
                return
            close_task = entry.close_task
            if close_task is None:
                close_task = asyncio.create_task(entry.client.close())
                entry.close_task = close_task
                close_task.add_done_callback(lambda task: self._finish_close(loop, entry, task))
        await asyncio.shield(close_task)

    def _finish_close(
        self,
        loop: asyncio.AbstractEventLoop,
        entry: _ClientEntry[_AsyncClientT],
        close_task: asyncio.Task[None],
    ) -> None:
        failed = close_task.cancelled()
        if not failed:
            failed = close_task.exception() is not None
        with self._lock:
            if self._clients.get(loop) is not entry or entry.close_task is not close_task:
                return
            if failed:
                entry.close_task = None
            else:
                self._clients.pop(loop)
