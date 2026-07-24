import asyncio
import atexit
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, List, Optional, TypeVar

from evalscope.utils.logger import get_logger

logger = get_logger()
T = TypeVar('T')


async def cancel_and_wait(task: 'asyncio.Task[Any]') -> None:
    """Cancel and observe a task while preserving cancellation of the caller."""
    if not task.done():
        task.cancel()
    waiter = asyncio.gather(task, return_exceptions=True)
    try:
        await asyncio.shield(waiter)
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        await waiter
        raise
    if not task.cancelled():
        task.result()


def shutdown_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Gracefully close an event loop that is no longer running."""
    if loop.is_closed():
        return

    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.run_until_complete(loop.shutdown_default_executor())
    loop.close()


class _LoopPhase(Enum):
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    CLOSED = 'closed'


@dataclass
class _LoopGeneration:
    loop: asyncio.AbstractEventLoop
    phase: _LoopPhase = _LoopPhase.STARTING
    ready: threading.Event = field(default_factory=threading.Event)
    stopped: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    close_callbacks: List[Callable[[], Awaitable[None]]] = field(default_factory=list)
    accepts_close_callbacks: bool = True
    shutdown_scheduled: bool = False
    shutdown_task: Optional[asyncio.Task[None]] = None
    callback_timeout: float = 5.0


# A delayed heartbeat identifies synchronous calls that block an owned loop.
_LOOP_HEALTH_ENABLED: bool = os.environ.get('EVALSCOPE_LOOP_HEALTH', '0') in ('1', 'true', 'TRUE')
_LOOP_HEALTH_INTERVAL_S: float = float(os.environ.get('EVALSCOPE_LOOP_HEALTH_INTERVAL', '0.5') or 0.5)
_LOOP_HEALTH_THRESHOLD_S: float = float(os.environ.get('EVALSCOPE_LOOP_HEALTH_THRESHOLD', '0.5') or 0.5)


def _install_loop_health_monitor(loop: asyncio.AbstractEventLoop, label: str) -> None:
    """Schedule a self-renewing heartbeat on ``loop`` to detect blocking."""
    state = {'last': time.monotonic()}

    def _tick() -> None:
        now = time.monotonic()
        delay = now - state['last'] - _LOOP_HEALTH_INTERVAL_S
        if delay > _LOOP_HEALTH_THRESHOLD_S:
            logger.warning(f'[loop-health] {label}: loop blocked for {delay * 1000:.0f}ms')
        state['last'] = now
        loop.call_later(_LOOP_HEALTH_INTERVAL_S, _tick)

    loop.call_later(_LOOP_HEALTH_INTERVAL_S, _tick)


class AsyncioLoopThread:
    """Long-lived asyncio event loop hosted by one daemon thread.

    The owning thread starts, runs, and closes the loop. Callers may submit
    coroutines synchronously or asynchronously from other threads and loops.
    """

    __slots__ = (
        '_name',
        '_health_label',
        '_state_lock',
        '_generation',
    )

    def __init__(self, name: str, *, health_label: Optional[str] = None) -> None:
        self._name = name
        self._health_label = health_label
        self._state_lock = threading.Lock()
        self._generation: Optional[_LoopGeneration] = None

    @property
    def active(self) -> bool:
        """Return whether the owned event loop is available."""
        with self._state_lock:
            generation = self._generation
            return generation is not None and generation.phase in (_LoopPhase.STARTING, _LoopPhase.RUNNING)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the owned event loop, starting its thread lazily."""
        return self._ensure_started()

    def owns(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Return whether ``loop`` is the currently owned event loop."""
        with self._state_lock:
            generation = self._generation
            return generation is not None and generation.loop is loop and generation.phase is not _LoopPhase.CLOSED

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._state_lock:
            return self._ensure_generation_locked().loop

    def _ensure_generation_locked(self) -> _LoopGeneration:
        generation = self._generation
        if generation is not None:
            if generation.phase is _LoopPhase.STOPPING:
                raise RuntimeError(f'{self._name} is stopping and cannot accept new work.')
            if generation.phase is not _LoopPhase.CLOSED:
                return generation

        loop = asyncio.new_event_loop()
        generation = _LoopGeneration(loop=loop)
        thread = threading.Thread(
            target=self._run_generation,
            args=(generation, ),
            daemon=True,
            name=self._name,
        )
        generation.thread = thread
        self._generation = generation
        thread.start()

        if _LOOP_HEALTH_ENABLED and self._health_label:
            loop.call_soon_threadsafe(_install_loop_health_monitor, loop, self._health_label)
        return generation

    def _run_generation(self, generation: _LoopGeneration) -> None:
        loop = generation.loop
        asyncio.set_event_loop(loop)
        try:
            with self._state_lock:
                if generation.phase is _LoopPhase.STARTING:
                    generation.phase = _LoopPhase.RUNNING
            generation.ready.set()
            loop.run_forever()
        finally:
            try:
                shutdown_event_loop(loop)
            finally:
                asyncio.set_event_loop(None)
                with self._state_lock:
                    generation.phase = _LoopPhase.CLOSED
                    generation.accepts_close_callbacks = False
                    if self._generation is generation:
                        self._generation = None
                generation.stopped.set()

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the owned loop and await it from any event loop."""
        try:
            running_loop = asyncio.get_running_loop()
            with self._state_lock:
                generation = self._ensure_generation_locked()
                run_on_current_loop = running_loop is generation.loop
                future = None if run_on_current_loop else asyncio.run_coroutine_threadsafe(coro, generation.loop)
        except BaseException:
            coro.close()
            raise
        if run_on_current_loop:
            return await coro
        assert future is not None
        return await asyncio.wrap_future(future)

    def submit(self, coro: Coroutine[Any, Any, T]) -> 'Future[T]':
        """Atomically admit and submit ``coro`` to the owned loop."""
        try:
            with self._state_lock:
                generation = self._ensure_generation_locked()
                return asyncio.run_coroutine_threadsafe(coro, generation.loop)
        except BaseException:
            coro.close()
            raise

    def run_sync(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """Run ``coro`` on the owned loop and block until it completes."""
        try:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            with self._state_lock:
                generation = self._ensure_generation_locked()
                if running_loop is generation.loop:
                    raise RuntimeError('Cannot synchronously wait on the owned event loop.')
                future = asyncio.run_coroutine_threadsafe(coro, generation.loop)
        except BaseException:
            coro.close()
            raise
        try:
            return future.result(timeout=timeout)
        except Exception:
            future.cancel()
            raise

    def add_close_callback(self, cb: Callable[[], Awaitable[None]]) -> bool:
        """Register an async cleanup callback to run before this loop stops."""
        with self._state_lock:
            generation = self._generation
            if generation is None or not generation.accepts_close_callbacks:
                return False
            generation.close_callbacks.append(cb)
            return True

    async def _run_close_callback_batch(
        self,
        generation: _LoopGeneration,
        *,
        close_admission_when_empty: bool,
    ) -> bool:
        with self._state_lock:
            callbacks = list(generation.close_callbacks)
            generation.close_callbacks.clear()
            if not callbacks and close_admission_when_empty:
                generation.accepts_close_callbacks = False
        if not callbacks:
            return False

        for cb in callbacks:
            try:
                await asyncio.wait_for(cb(), timeout=generation.callback_timeout)
            except asyncio.CancelledError:
                logger.warning('loop close callback was cancelled')
            except Exception as e:
                logger.warning(f'loop close callback failed: {e}')
        return True

    async def _cancel_application_tasks(self) -> None:
        current_task = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current_task and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _shutdown_generation(self, generation: _LoopGeneration) -> None:
        try:
            await self._run_close_callback_batch(generation, close_admission_when_empty=False)
            await self._cancel_application_tasks()
            while await self._run_close_callback_batch(generation, close_admission_when_empty=True):
                pass
        finally:
            generation.loop.stop()

    def _schedule_shutdown(self, generation: _LoopGeneration) -> None:

        def _start_shutdown() -> None:
            if generation.shutdown_task is None:
                generation.shutdown_task = asyncio.create_task(self._shutdown_generation(generation))

        try:
            generation.loop.call_soon_threadsafe(_start_shutdown)
        except RuntimeError:
            pass

    def stop(self, join_timeout: float = 5.0) -> bool:
        """Run cleanup callbacks, stop the loop, and join its thread."""
        with self._state_lock:
            generation = self._generation
            if generation is None or generation.phase is _LoopPhase.CLOSED:
                return True
            first_stop = not generation.shutdown_scheduled
            if first_stop:
                generation.phase = _LoopPhase.STOPPING
                generation.shutdown_scheduled = True
                generation.callback_timeout = join_timeout
            thread = generation.thread
            callback_count = len(generation.close_callbacks)

        if first_stop:
            generation.ready.wait(timeout=join_timeout)
            self._schedule_shutdown(generation)

        if thread is threading.current_thread():
            return False

        wait_timeout = join_timeout * (callback_count + 2)
        stopped = generation.stopped.wait(timeout=wait_timeout)
        if not stopped:
            logger.warning(f'{self._name} did not stop within {wait_timeout} seconds')
        return stopped


class AsyncioLoopRunner:
    """Per-thread asyncio loop runner for sync-to-async bridging."""

    _local = threading.local()
    _all_handles_lock = threading.Lock()
    _all_handles: List[AsyncioLoopThread] = []

    @classmethod
    def _get_handle(cls) -> AsyncioLoopThread:
        handle: Optional[AsyncioLoopThread] = getattr(cls._local, 'handle', None)
        if handle is not None:
            return handle
        owner_name = threading.current_thread().name
        handle = AsyncioLoopThread(name=f'EvalLoop[{owner_name}]', health_label=owner_name)
        cls._local.handle = handle
        with cls._all_handles_lock:
            cls._all_handles.append(handle)
        return handle

    @classmethod
    def run(cls, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """Submit a coroutine to this thread's owned loop and wait for its result."""
        handle = cls._get_handle()
        return handle.run_sync(coro, timeout=timeout)

    @classmethod
    def shutdown_for_thread(cls, join_timeout: float = 5.0) -> None:
        """Stop and release the loop bound to the calling thread, if any."""
        handle: Optional[AsyncioLoopThread] = getattr(cls._local, 'handle', None)
        if handle is None:
            return
        if not handle.stop(join_timeout=join_timeout):
            return
        cls._local.handle = None
        with cls._all_handles_lock:
            try:
                cls._all_handles.remove(handle)
            except ValueError:
                pass

    @classmethod
    def shutdown_all(cls, join_timeout: float = 5.0) -> None:
        """Stop every loop created by this runner."""
        with cls._all_handles_lock:
            handles = list(cls._all_handles)
        # Only the owner thread can clear its thread-local handle. Keep stopped
        # handles registered in case a worker thread later starts a generation.
        for handle in handles:
            handle.stop(join_timeout=join_timeout)

    @classmethod
    def register_close_callback(cls, cb: Callable[[], Awaitable[None]]) -> bool:
        """Register cleanup on the runner-owned loop currently executing."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return False
        with cls._all_handles_lock:
            for handle in cls._all_handles:
                if handle.owns(running):
                    return handle.add_close_callback(cb)
        return False


atexit.register(AsyncioLoopRunner.shutdown_all)
