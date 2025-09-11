import threading
import time
from contextlib import contextmanager
from functools import wraps


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


def thread_safe(func):
    """Thread-safe decorator for functions that need to be executed in a thread-safe manner."""
    lock = threading.RLock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

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
