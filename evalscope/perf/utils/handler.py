import asyncio
import functools
import os
import platform
import signal
import sys

from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Event loop policy: optional uvloop
# ---------------------------------------------------------------------------

# Module-level flag so we only attempt to install uvloop once per process,
# even when ``run_one_benchmark`` is invoked repeatedly from a sweep.
_UVLOOP_INSTALL_ATTEMPTED = False


def install_uvloop_if_available() -> None:
    """Best-effort enable uvloop as the asyncio event loop policy.

    Why this exists
    ---------------
    The default CPython selector loop has visible scheduling jitter under
    high-concurrency LLM benchmarking (many concurrent SSE streams + bursty
    chunk callbacks contending for the same loop tick).  That jitter shows
    up as a small but persistent shortfall between the configured request
    rate (``--rate``) and the rate actually realised by the dispatcher.
    uvloop is a libuv-backed loop that drives ``asyncio.sleep`` and I/O
    callbacks with substantially higher precision and throughput, which
    keeps the realised QPS closer to the target.

    Behaviour
    ---------
    * Skipped on Windows (uvloop has no Windows support; the existing
      ``WindowsSelectorEventLoopPolicy`` branch below stays intact).
    * Skipped if uvloop is not installed -- evalscope continues to work
      with the default loop, just with slightly looser rate control.
    * Can be force-disabled by setting ``EVALSCOPE_DISABLE_UVLOOP=1`` as
      an escape hatch for environments where uvloop misbehaves.
    * Idempotent: only attempts the install once per process.
    """
    global _UVLOOP_INSTALL_ATTEMPTED
    if _UVLOOP_INSTALL_ATTEMPTED:
        return
    _UVLOOP_INSTALL_ATTEMPTED = True

    if platform.system() == 'Windows':
        return
    if os.environ.get('EVALSCOPE_DISABLE_UVLOOP', '').strip() in ('1', 'true', 'True'):
        logger.info('uvloop disabled via EVALSCOPE_DISABLE_UVLOOP; using default asyncio loop')
        return

    try:
        import uvloop  # type: ignore
    except ImportError:
        logger.info(
            'uvloop not installed; using default asyncio loop. '
            'Install with `pip install uvloop` (or `pip install evalscope[perf]`) '
            'for tighter rate control under high concurrency.'
        )
        return

    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info('uvloop event loop policy installed (asyncio.sleep precision improved)')
    except Exception as e:  # noqa: BLE001 -- never let event-loop choice break a run
        logger.warning(f'Failed to install uvloop policy ({e}); falling back to default asyncio loop')


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


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
