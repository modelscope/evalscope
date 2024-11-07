import asyncio
import functools
import sys

from evalscope.perf.utils._logging import logger


def exception_handler(func):
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in async function '{func.__name__}': {e}")
                sys.exit(1)

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in function '{func.__name__}': {e}")
                sys.exit(1)

        return sync_wrapper
