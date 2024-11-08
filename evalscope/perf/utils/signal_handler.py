import functools
import signal

from evalscope.utils.logger import get_logger

logger = get_logger()


def signal_handler(signal_name, loop):
    logger.info('Got signal %s: exit' % signal_name)
    loop.stop()


def add_signal_handlers(loop):
    for signal_name in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signal_name),
            functools.partial(signal_handler, signal_name, loop),
        )
