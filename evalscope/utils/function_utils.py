import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from functools import wraps
from tqdm import tqdm
from typing import Any, Awaitable, Callable, List, Optional, Sequence, TypeVar, Union

from evalscope.utils.logger import get_logger

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


def retry_wrapper(retries=3, sleep_interval=0):
    """
    Decorator that retries a function call up to `retries` times if an exception occurs.

    Args:
        retries: Maximum number of retry attempts (default: 3)
        sleep_interval: Seconds to wait between retries (default: 0)

    Returns:
        Decorated function that implements retry logic with logging
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
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

        return wrapper

    return decorator


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


class AsyncioLoopRunner:
    """Singleton background asyncio loop runner for sync→async bridging."""
    _instance: Optional['AsyncioLoopRunner'] = None
    _inst_lock = threading.Lock()

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._start_loop()

    def _start_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True, name='AsyncioLoopRunner')
        self._thread.start()

    @classmethod
    def instance(cls) -> 'AsyncioLoopRunner':
        if cls._instance is not None:
            return cls._instance
        with cls._inst_lock:
            if cls._instance is None:
                cls._instance = AsyncioLoopRunner()
        return cls._instance

    @classmethod
    def run(cls, coro: Awaitable[T], timeout: Optional[float] = None) -> T:
        """Submit a coroutine to the background loop and wait for result."""
        inst = cls.instance()
        fut = asyncio.run_coroutine_threadsafe(coro, inst._loop)
        return fut.result(timeout=timeout)

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Access the underlying event loop (read-only use)."""
        return self._loop

    def stop(self, join_timeout: float = 5.0) -> None:
        """Optional shutdown of the background loop (generally not needed)."""
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=join_timeout)


def run_in_threads_with_progress(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    desc: str,
    max_workers: int,
    heartbeat_sec: int,
    on_result: Optional[Callable[[T, R], None]] = None,
    on_error: Optional[Callable[[T, Exception], None]] = None,
    filter_none_results: bool = False,
) -> List[R]:
    """
    Execute a collection of tasks concurrently with a ThreadPoolExecutor while
    displaying a tqdm progress bar and emitting periodic heartbeat logs.

    Key behaviors:
    - Concurrency: Uses up to `min(len(items), max_workers)` threads.
    - Progress: A tqdm bar advances when each task finishes (success or failure).
    - Heartbeat: If no tasks finish within `heartbeat_sec`, a status line is logged.
    - Ordering: Results are appended in completion order (not the original order).
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
        heartbeat_sec: Interval (in seconds) to wait before emitting a heartbeat log if
            no tasks complete in that window.
        on_result: Optional callback invoked as on_result(item, result) after success.
        on_error: Optional callback invoked as on_error(item, exception) on failure. If omitted,
            the exception is propagated and the function terminates early.

    Returns:
        A list of results collected as tasks complete (completion order).
        If some tasks fail and `on_error` is provided (and does not re-raise), those failures
        are skipped and not included in the returned results.

    Raises:
        Exception: Propagates the first task exception if `on_error` is not provided, or if
        `on_error` re-raises.

    Notes:
        - The function is blocking until all tasks complete or an exception is propagated.
        - Use `on_error` to implement "best-effort" processing where failures are logged
          and the rest continue.
    """
    # Defensive copy to avoid consuming a generator multiple times and to compute pool size.
    pending_items: List[T] = list(items)
    if not pending_items:
        return []

    # Include indices to ensure results are returned in input order
    indexed_items = list(enumerate(items))
    results: List[Optional[R]] = [None] * len(items)  # Preallocate results list

    # Bound the pool by actual workload size for efficiency.
    with ThreadPoolExecutor(max_workers=min(len(indexed_items), max_workers)) as executor:
        # Submit all tasks up-front and map futures back to their originating item.
        future_to_index = {executor.submit(worker, item): index for index, item in indexed_items}

        # Progress bar reflects total number of submitted tasks; updated per finished future.
        with tqdm(total=len(indexed_items), desc=desc, mininterval=1, dynamic_ncols=True) as pbar:
            # Track unfinished futures and poll with a timeout to enable heartbeat logs.
            pending = set(future_to_index.keys())
            while pending:
                # Wait with timeout to detect stalls and emit heartbeats proactively.
                done, not_done = wait(pending, timeout=heartbeat_sec)
                if not done:
                    # Heartbeat when nothing has completed within the window.
                    logger.info(f'{desc} still processing... pending={len(not_done)}')
                    continue

                # Consume completed futures.
                for future in done:
                    index = future_to_index[future]
                    try:
                        res = future.result()
                        results[index] = res  # Store result at the correct index
                        # Invoke success callback in caller thread (not in worker).
                        if on_result is not None:
                            on_result(items[index], res)
                    except Exception as exc:
                        # Delegate failure handling to on_error if provided; otherwise bubble up.
                        if on_error is not None:
                            on_error(items[index], exc)
                        else:
                            raise
                    finally:
                        # Always advance progress for completed futures (success or failure).
                        pbar.update(1)
                        pbar.refresh()

                # Continue polling remaining futures.
                pending = not_done

    # Return results, which are now guaranteed to be in input order
    if filter_none_results:
        # Filter out None results if on_error was used and some tasks failed
        results = [res for res in results if res is not None]
    return results
