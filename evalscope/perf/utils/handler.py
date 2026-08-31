import asyncio
import functools
import inspect
import os
import platform
import signal
from dataclasses import dataclass
from typing import Optional

from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Event loop policy: optional uvloop
# ---------------------------------------------------------------------------

# Module-level flag so we only attempt to install uvloop once per process,
# even when ``run_one_benchmark`` is invoked repeatedly from a sweep.
_UVLOOP_INSTALL_ATTEMPTED = False


@dataclass
class ShutdownSignalState:
    """Signal received by a benchmark loop, if any."""

    signal_name: Optional[str] = None

    @property
    def exit_code(self) -> int:
        """Return the conventional shell exit code for the received signal."""
        if self.signal_name is None:
            raise RuntimeError('No shutdown signal has been received.')
        return 128 + getattr(signal, self.signal_name)


class PerfBenchmarkInterrupted(Exception):
    """Raised after a signal-triggered benchmark cancellation finishes cleanup."""

    def __init__(self, signal_state: ShutdownSignalState) -> None:
        self.signal_name = signal_state.signal_name
        self.exit_code = signal_state.exit_code
        super().__init__(f'Benchmark interrupted by {self.signal_name}')


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
    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def async_generator_wrapper(*args, **kwargs):
            try:
                async for item in func(*args, **kwargs):
                    yield item
            except Exception as e:
                logger.exception(f"Exception in async generator '{func.__name__}': {e}")
                raise

        return async_generator_wrapper

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in async function '{func.__name__}': {e}")
                raise

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in function '{func.__name__}': {e}")
                raise

        return sync_wrapper


def signal_handler(
    signal_name: str,
    loop: asyncio.AbstractEventLoop,
    signal_state: Optional[ShutdownSignalState] = None,
) -> None:
    """Gracefully interrupt a running benchmark loop.

    ``loop.stop()`` aborts the loop mid-flight, which surfaces from
    ``run_until_complete`` as a confusing ``RuntimeError: Event loop stopped
    before Future completed`` and skips every ``finally`` block of the running
    coroutine (request teardown, DB cleanup).  Cancelling the pending tasks
    instead delivers ``CancelledError`` to the benchmark coroutine, so cleanup
    runs and ``run_until_complete`` unwinds normally.
    """
    if signal_state is not None:
        signal_state.signal_name = signal_name
    logger.info(f'Got signal {signal_name}: cancelling pending tasks')
    for task in asyncio.all_tasks(loop):
        task.cancel()


def add_signal_handlers(loop: asyncio.AbstractEventLoop) -> ShutdownSignalState:
    signal_state = ShutdownSignalState()
    for signal_name in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signal_name),
            functools.partial(signal_handler, signal_name, loop, signal_state),
        )
    return signal_state
