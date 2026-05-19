import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, TypeVar, Union

from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

logger = get_logger()

T = TypeVar('T')
R = TypeVar('R')

# Global lock to safely create per-instance locks in decorators
_THREAD_SAFE_GLOBAL_LOCK = threading.RLock()


def thread_safe(func: Callable[..., T]) -> Callable[..., T]:
    """Thread-safe decorator.
    - If decorating a bound method, uses a per-instance, per-method lock.
    - If decorating a function, uses a function-scoped lock.
    """
    func_lock = threading.RLock()
    lock_attr_name = f'__lock_{func.__name__}'

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Prefer per-instance lock if the first arg looks like 'self'
        if args and hasattr(args[0], '__dict__'):
            self_obj = args[0]
            lock = getattr(self_obj, lock_attr_name, None)
            if lock is None:
                with _THREAD_SAFE_GLOBAL_LOCK:
                    lock = getattr(self_obj, lock_attr_name, None)
                    if lock is None:
                        lock = threading.RLock()
                        setattr(self_obj, lock_attr_name, lock)
        else:
            lock = func_lock

        with lock:
            return func(*args, **kwargs)

    return wrapper


def run_once(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure a function is executed at most once across threads."""
    lock = threading.RLock()
    has_run: bool = False
    result: Optional[T] = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run, result
        if has_run:
            return result
        # Double-checked locking to avoid redundant locking on hot path
        with lock:
            if not has_run:
                result = func(*args, **kwargs)
                has_run = True
        return result

    return wrapper


def retry_call(func, *args, retries=3, sleep_interval=0, **kwargs):
    """Function that retries a function call up to `retries` times if an exception occurs."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                if sleep_interval > 0:
                    logger.warning(f'Attempt {attempt + 1} / {retries} failed: {e}. Retrying...')
                    time.sleep(sleep_interval)
            else:
                raise


async def async_retry_call(
    func: Callable[..., Awaitable[T]], *args, retries: int = 3, sleep_interval: float = 0, **kwargs
) -> T:
    """Async version of retry_call. Retries an async function call up to `retries` times if an exception occurs."""
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                if sleep_interval > 0:
                    logger.warning(f'Attempt {attempt + 1} / {retries} failed: {e}. Retrying...')
                    await asyncio.sleep(sleep_interval)
            else:
                raise


class _LoopHandle:
    """One asyncio loop running on its own daemon thread.

    Owned by ``AsyncioLoopRunner`` and stored thread-locally so each
    calling worker thread gets its own isolated event loop.
    """

    __slots__ = ('loop', 'thread', '_health_task', '_close_callbacks', '_close_callbacks_lock')

    def __init__(self, owner_thread_name: str) -> None:
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._health_task: Optional[asyncio.Handle] = None
        # Async cleanup callbacks to run on this loop just before it stops.
        # Used by per-loop resource caches (e.g. AsyncOpenAI / AsyncAnthropic
        # clients) to close their httpx connection pools while the loop they
        # were bound to is still alive.
        self._close_callbacks: List[Callable[[], Awaitable[None]]] = []
        self._close_callbacks_lock = threading.Lock()

        def _run() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(
            target=_run,
            daemon=True,
            name=f'EvalLoop[{owner_thread_name}]',
        )
        self.thread.start()

        if _LOOP_HEALTH_ENABLED:
            self.loop.call_soon_threadsafe(_install_loop_health_monitor, self.loop, owner_thread_name)

    def add_close_callback(self, cb: Callable[[], Awaitable[None]]) -> None:
        """Register an async cleanup callback to run before this loop stops."""
        with self._close_callbacks_lock:
            self._close_callbacks.append(cb)

    def _run_close_callbacks(self, per_cb_timeout: float) -> None:
        """Run registered close callbacks on this loop. Runs from external thread."""
        with self._close_callbacks_lock:
            callbacks = list(self._close_callbacks)
            self._close_callbacks.clear()
        for cb in callbacks:
            try:
                fut = asyncio.run_coroutine_threadsafe(cb(), self.loop)
                fut.result(timeout=per_cb_timeout)
            except Exception as e:
                logger.warning(f'loop close callback failed: {e}')

    def stop(self, join_timeout: float = 5.0) -> None:
        # Drain async cleanup hooks while the loop is still running, so
        # resources like httpx pools tied to this loop get closed properly.
        if self.loop.is_running():
            self._run_close_callbacks(per_cb_timeout=join_timeout)
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        self.thread.join(timeout=join_timeout)
        try:
            self.loop.close()
        except Exception:
            pass


# Health monitor: schedules a periodic call_soon and measures the wall-clock
# delay between scheduling and execution. A delay > _LOOP_HEALTH_THRESHOLD_S
# indicates the loop is being blocked by a sync call, which is the exact
# bug class this runner refactor is designed to surface.
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


class AsyncioLoopRunner:
    """Per-thread asyncio loop runner for sync→async bridging.

    Each calling thread that invokes :meth:`run` gets its own dedicated
    asyncio event loop running on its own daemon thread. This isolates
    samples from each other: a sync block in one sample's coroutine cannot
    pause another sample's loop.

    Use :meth:`shutdown_for_thread` from the worker thread when it's done
    (e.g. in a ``finally`` block at the end of a worker function) to release
    the loop and its background thread.
    """

    _local = threading.local()
    _all_handles_lock = threading.Lock()
    _all_handles: List[_LoopHandle] = []  # for atexit cleanup

    @classmethod
    def _get_handle(cls) -> _LoopHandle:
        handle: Optional[_LoopHandle] = getattr(cls._local, 'handle', None)
        if handle is not None:
            return handle
        owner_name = threading.current_thread().name
        handle = _LoopHandle(owner_name)
        cls._local.handle = handle
        with cls._all_handles_lock:
            cls._all_handles.append(handle)
        return handle

    @classmethod
    def run(cls, coro: Awaitable[T], timeout: Optional[float] = None) -> T:
        """Submit a coroutine to *this thread's* loop and wait for result."""
        handle = cls._get_handle()
        fut = asyncio.run_coroutine_threadsafe(coro, handle.loop)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            # On timeout/cancel, request the running coroutine to cancel as
            # well so it doesn't keep running on the background loop.
            try:
                fut.cancel()
            except Exception:
                pass
            raise

    @classmethod
    def shutdown_for_thread(cls, join_timeout: float = 5.0) -> None:
        """Stop and release the loop bound to the calling thread (if any).

        Idempotent. Worker threads that call :meth:`run` should invoke this
        in a ``finally`` block to avoid leaking a daemon thread + loop per
        worker.
        """
        handle: Optional[_LoopHandle] = getattr(cls._local, 'handle', None)
        if handle is None:
            return
        handle.stop(join_timeout=join_timeout)
        cls._local.handle = None
        with cls._all_handles_lock:
            try:
                cls._all_handles.remove(handle)
            except ValueError:
                pass

    @classmethod
    def shutdown_all(cls, join_timeout: float = 5.0) -> None:
        """Stop every loop ever created by this runner. For atexit only."""
        with cls._all_handles_lock:
            handles = list(cls._all_handles)
            cls._all_handles.clear()
        for h in handles:
            h.stop(join_timeout=join_timeout)

    @classmethod
    def register_close_callback(cls, cb: Callable[[], Awaitable[None]]) -> bool:
        """Register an async cleanup callback on the currently running loop.

        Intended to be called from inside a coroutine running on a runner-owned
        loop. The callback is awaited on that loop just before it stops, so
        resources bound to it (e.g. AsyncOpenAI / AsyncAnthropic httpx pools)
        can close cleanly. Returns True if the running loop is one of ours and
        the callback was registered.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return False
        with cls._all_handles_lock:
            for handle in cls._all_handles:
                if handle.loop is running:
                    handle.add_close_callback(cb)
                    return True
        return False

    @classmethod
    def loop_for_current_thread(cls) -> Optional[asyncio.AbstractEventLoop]:
        """Read-only access to the calling thread's loop, if one exists."""
        handle: Optional[_LoopHandle] = getattr(cls._local, 'handle', None)
        return handle.loop if handle is not None else None


# Process-level safety net: if any per-thread loops are still alive at
# interpreter exit (typically because a worker thread exited without
# calling shutdown_for_thread), tear them down so we don't leak file
# descriptors (httpx connection pools, sandbox sockets, etc.).
import atexit as _atexit  # noqa: E402

_atexit.register(AsyncioLoopRunner.shutdown_all)


def run_in_threads_with_progress(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    desc: str,
    max_workers: int,
    log_interval: Optional[int] = None,
    on_result: Optional[Callable[[T, R], None]] = None,
    on_error: Optional[Callable[[T, Exception], None]] = None,
    skip_failed: bool = False,
    initial: int = 0,
    total: Optional[int] = None,
) -> List[R]:
    """
    Execute a collection of tasks concurrently with a ThreadPoolExecutor while
    displaying a tqdm progress bar and emitting periodic heartbeat logs.

    Key behaviors:
    - Concurrency: Uses up to `min(len(items), max_workers)` threads.
    - Progress: A tqdm bar advances when each task finishes (success or failure).
    - Heartbeat: If no tasks finish within `log_interval` seconds, a status line is logged.
    - Ordering: Results preserve the original input order.
    - Error handling:
        * If `on_error` is provided, it is called for each failed item; execution continues
          unless `on_error` itself raises.
        * If `on_error` is None, the first exception is raised immediately and stops processing.
    - Callbacks:
        * `on_result(item, result)` is called after a successful result is obtained.
        * Both callbacks run in the main thread (not worker threads).

    Args:
        items: A sequence of items (inputs) to process. Converted to a list internally.
        worker: A callable executed in threads to process a single item and return a result.
        desc: A short text shown as the tqdm progress bar description.
        max_workers: Upper bound on the number of concurrent threads.
        log_interval: Interval (in seconds) to wait before emitting a heartbeat log if
            no tasks complete in that window.
        on_result: Optional callback invoked as on_result(item, result) after success.
        on_error: Optional callback invoked as on_error(item, exception) on failure. If omitted,
            the exception is propagated and the function terminates early.
        skip_failed: If True, items whose tasks raised an exception (and were handled by
            `on_error`) are omitted from the returned list. Only meaningful when `on_error`
            is provided; otherwise exceptions are always propagated.
        initial: Number of already-completed items (e.g. loaded from cache). The progress
            bar will start at this offset so the full work is visible.
        total: Override the displayed total. Defaults to ``initial + len(items)``.

    Returns:
        A list of results in the original input order.
        If some tasks fail and `on_error` is provided (and does not re-raise), those slots
        are omitted when `skip_failed=True`, otherwise they appear as ``None``.

    Raises:
        Exception: Propagates the first task exception if `on_error` is not provided, or if
        `on_error` re-raises.

    Notes:
        - The function is blocking until all tasks complete or an exception is propagated.
        - Use `on_error` to implement "best-effort" processing where failures are logged
          and the rest continue.
    """
    indexed_work_items = list(enumerate(items))
    ordered_results: List[Optional[R]] = [None] * len(items)  # pre-allocated; preserves input order

    # Resolve progress-bar total: default to initial + actual workload size
    progress_bar_total = total if total is not None else initial + len(indexed_work_items)

    # Bound max_workers to actual workload size to avoid unnecessarily large thread pools
    effective_max_workers = max(1, min(max_workers, len(indexed_work_items)))

    # Wrap the worker so the per-thread asyncio loop (if any was lazily created
    # by AsyncioLoopRunner during the worker call) is shut down when the
    # worker returns. This avoids leaking one daemon thread + one event loop
    # per ThreadPoolExecutor worker across the eval lifetime.
    def _worker_with_loop_cleanup(item: T) -> R:
        try:
            return worker(item)
        finally:
            AsyncioLoopRunner.shutdown_for_thread()

    with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
        with tqdm(
            total=progress_bar_total,
            initial=initial,
            desc=desc,
            mininterval=1,
            dynamic_ncols=True,
            logger=logger,
            log_interval=log_interval,
            track_progress=True,
        ) as pbar:
            # Submit tasks in a sliding window of size max_workers so that an early
            # failure does not leave many already-submitted futures still running.
            item_iter = iter(indexed_work_items)
            pending_futures: set = set()
            future_index_map: dict = {}

            def _try_submit_next():
                """Submit the next item from the iterator into the thread pool, if any remain."""
                try:
                    item_index, item = next(item_iter)
                    fut = executor.submit(_worker_with_loop_cleanup, item)
                    future_index_map[fut] = item_index
                    pending_futures.add(fut)
                except StopIteration:
                    pass

            # Fill the initial sliding window.
            for _ in range(max_workers):
                _try_submit_next()

            while pending_futures:
                completed_futures, _ = wait(pending_futures, timeout=1)
                if not completed_futures:
                    pbar.check_log()
                    continue

                for fut in completed_futures:
                    item_index = future_index_map.pop(fut)
                    pending_futures.discard(fut)
                    try:
                        result = fut.result()
                        ordered_results[item_index] = result
                        if on_result is not None:
                            on_result(items[item_index], result)
                        _try_submit_next()  # keep the window full on success
                    except Exception as error:
                        if on_error is not None:
                            on_error(items[item_index], error)  # may re-raise
                            _try_submit_next()  # error was tolerated, keep going
                        else:
                            raise
                    finally:
                        pbar.update(1)

    # Return results in input order; optionally compact out failed (None) slots
    if skip_failed:
        return [res for res in ordered_results if res is not None]
    return ordered_results
