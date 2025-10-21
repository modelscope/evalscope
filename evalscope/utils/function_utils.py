import asyncio
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar('T')

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


def run_once(func):
    """Decorator to ensure a function is only run once."""
    has_run = False
    result = None

    def wrapper(*args, **kwargs):
        nonlocal has_run, result
        if not has_run:
            result = func(*args, **kwargs)
            has_run = True
        return result

    return wrapper


def retry_func(retries=3, sleep_interval=0):
    """A decorator that retries a function call up to `retries` times if an exception occurs."""

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if sleep_interval > 0:
                        time.sleep(sleep_interval)
            raise last_exception

        return wrapper

    return decorator


@contextmanager
def retry_context(retries=3, sleep_interval=0):
    """A context manager that retries the code block up to `retries` times if an exception occurs."""
    last_exception = None
    for attempt in range(retries):
        try:
            yield
            return  # If no exception, exit successfully
        except Exception as e:
            last_exception = e
            if sleep_interval > 0:
                time.sleep(sleep_interval)
            if attempt == retries - 1:  # Last attempt
                break
    raise last_exception


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
