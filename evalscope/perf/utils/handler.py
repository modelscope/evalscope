import asyncio
import functools
import signal
import sys

from evalscope.utils.logger import get_logger

logger = get_logger()


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


def signal_handler(signal_name, loop):
    logger.info('Got signal %s: exit' % signal_name)
    loop.stop()


def add_signal_handlers(loop):
    for signal_name in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signal_name),
            functools.partial(signal_handler, signal_name, loop),
        )
