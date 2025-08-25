import threading
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
