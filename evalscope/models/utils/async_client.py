import asyncio
import threading
from typing import Callable, Dict, Generic, List, Protocol, TypeVar

from evalscope.utils.function_utils import AsyncioLoopRunner


class _AsyncCloseable(Protocol):

    async def close(self) -> None:
        ...


_AsyncClientT = TypeVar('_AsyncClientT', bound=_AsyncCloseable)


class LoopBoundAsyncClientPool(Generic[_AsyncClientT]):
    """Own one async client per event loop and close each on its owner loop."""

    def __init__(self, factory: Callable[[], _AsyncClientT]) -> None:
        self._factory = factory
        self._clients: Dict[asyncio.AbstractEventLoop, _AsyncClientT] = {}
        self._lock = threading.Lock()

    def get(self) -> _AsyncClientT:
        """Return the client associated with the running event loop."""
        loop = asyncio.get_running_loop()
        with self._lock:
            client = self._clients.get(loop)
            if client is None:
                client = self._factory()
                self._clients[loop] = client
                AsyncioLoopRunner.register_close_callback(lambda: self._close_on_owner_loop(loop, client))
            return client

    async def aclose(self) -> None:
        """Close all cached clients on their respective owner loops."""
        with self._lock:
            clients = list(self._clients.items())

        errors: List[Exception] = []
        for loop, client in clients:
            try:
                await self._close_client(loop, client)
            except Exception as e:
                errors.append(e)

        if errors:
            raise errors[0]

    async def _close_client(self, loop: asyncio.AbstractEventLoop, client: _AsyncClientT) -> None:
        running_loop = asyncio.get_running_loop()
        if loop is running_loop:
            await self._close_on_owner_loop(loop, client)
            return

        if loop.is_closed() or not loop.is_running():
            self._discard(loop, client)
            raise RuntimeError(
                'Cannot close an async model client after its owner event loop has stopped. '
                'Call ModelAPI.aclose() before shutting down the loop.'
            )

        future = asyncio.run_coroutine_threadsafe(self._close_on_owner_loop(loop, client), loop)
        await asyncio.wrap_future(future)

    async def _close_on_owner_loop(self, loop: asyncio.AbstractEventLoop, client: _AsyncClientT) -> None:
        if not self._take(loop, client):
            return
        try:
            await client.close()
        except Exception:
            with self._lock:
                self._clients.setdefault(loop, client)
            raise

    def _take(self, loop: asyncio.AbstractEventLoop, client: _AsyncClientT) -> bool:
        with self._lock:
            if self._clients.get(loop) is not client:
                return False
            self._clients.pop(loop)
            return True

    def _discard(self, loop: asyncio.AbstractEventLoop, client: _AsyncClientT) -> None:
        with self._lock:
            if self._clients.get(loop) is client:
                self._clients.pop(loop)
