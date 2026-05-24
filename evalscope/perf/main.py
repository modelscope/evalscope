import asyncio
import copy
import os
import platform
import threading
import time
from argparse import Namespace

from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.utils.logger import configure_logging, get_logger
from evalscope.utils.model_utils import seed_everything
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm
from evalscope.utils.tqdm_utils import make_tracker
from .arguments import Arguments, parse_args
from .benchmark import run_benchmark
from .multi_turn_benchmark import run_multi_turn_benchmark
from .sla.sla_run import run_sla_auto_tune
from .utils.db_util import get_output_path
from .utils.handler import add_signal_handlers
from .utils.local_server import start_app
from .utils.log_utils import init_visualizer
from .utils.report.generate_report import gen_perf_html_report
from .utils.rich_display import print_summary

logger = get_logger()

# Module-level flag so we only attempt to install uvloop once per process,
# even when ``run_one_benchmark`` is invoked repeatedly from a sweep
# (see ``run_multi_benchmark``).
_UVLOOP_INSTALL_ATTEMPTED = False


def _install_uvloop_if_available() -> None:
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


def run_one_benchmark(args: Arguments, output_path: str = None):
    """Run a single benchmark with given parallel/rate and number settings."""
    if isinstance(args.parallel, list):
        args.parallel = args.parallel[0]
    if isinstance(args.number, list):
        args.number = args.number[0]
    if isinstance(args.rate, list):
        args.rate = args.rate[0]

    logger.info('Starting benchmark with args: ')
    logger.info(args)

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
        # Try to upgrade to uvloop on POSIX systems for higher-precision
        # ``asyncio.sleep`` / I/O dispatch.  No-op on failure.
        _install_uvloop_if_available()

    loop = asyncio.new_event_loop()
    # Only add signal handlers in main thread
    if platform.system() != 'Windows' and threading.current_thread() is threading.main_thread():
        try:
            add_signal_handlers(loop)
        except ValueError as e:
            logger.warning(f'Cannot add signal handlers (running in non-main thread): {e}')

    with args.output_context(output_path):
        if args.multi_turn:
            metrics_result, percentile_result = loop.run_until_complete(run_multi_turn_benchmark(args))
        else:
            metrics_result, percentile_result = loop.run_until_complete(run_benchmark(args))

    # Return unified format; key reflects the sweep dimension
    if args.open_loop:
        key = f'rate_{args.rate}_number_{args.number}'
    else:
        key = f'parallel_{args.parallel}_number_{args.number}'
    return {key: {'metrics': metrics_result, 'percentiles': percentile_result}}


def run_multi_benchmark(args: Arguments, output_path: str = None):
    """Run multiple benchmarks with different parallel/rate and number combinations."""
    results = {}
    number_list = copy.deepcopy(args.number)

    if args.open_loop:
        sweep_attr = 'rate'
        sweep_list = copy.deepcopy(args.rate)
        run_name_fmt = 'rate_{sweep}_number_{number}'
        desc = 'Running[perf-open-loop]'
    else:
        sweep_attr = 'parallel'
        sweep_list = copy.deepcopy(args.parallel)
        run_name_fmt = 'parallel_{sweep}_number_{number}'
        desc = 'Running[perf]'

    total = len(number_list)
    pbar = tqdm(
        enumerate(zip(number_list, sweep_list)),
        total=total,
        desc=desc,
        logger=logger,
        log_interval=HEARTBEAT_INTERVAL_SEC,
    )
    for i, (number, sweep_val) in pbar:
        args.number = number
        setattr(args, sweep_attr, sweep_val)

        cur_run_name = run_name_fmt.format(sweep=sweep_val, number=number)
        cur_output_path = os.path.join(output_path, cur_run_name)
        os.makedirs(cur_output_path, exist_ok=True)

        benchmark_result = run_one_benchmark(args, output_path=cur_output_path)
        results.update(benchmark_result)

        # Auto-advance dataset_offset so the next run starts at a different position
        # in the token vocabulary, preventing KV-cache hits from identical prompts.
        args.dataset_offset += number

        if i < total - 1:
            logger.info(f'Sleeping for {args.sleep_interval} seconds before the next run...')
            time.sleep(args.sleep_interval)

    print_summary(results, args)

    return results


def run_perf_benchmark(args):
    """
    Run performance benchmark with given arguments.

    Args:
        args: Arguments object, dict, or Namespace

    Returns:
        Dict with benchmark results in format {parallel_x_number_x: {metrics: ..., percentiles: ...}}
    """
    # Check if args is a dictionary or Namespace
    if isinstance(args, dict):
        args = Arguments(**args)
    elif isinstance(args, Namespace):
        args = Arguments.from_args(args)

    if args.seed is not None:
        seed_everything(args.seed)

    # Initialize output directory
    output_path = get_output_path(args)
    configure_logging(args.debug, os.path.join(output_path, 'benchmark.log'))
    args.outputs_dir = output_path

    if args.sla_auto_tune:
        results = run_sla_auto_tune(args, run_one_benchmark)
        return results

    # Initialize visualizer
    init_visualizer(args)

    # Initialize local server if needed
    if args.api.startswith('local'):
        #  start local server
        server = threading.Thread(target=start_app, args=(copy.deepcopy(args), ), daemon=True)
        server.start()

    total_count = sum(args.number) if isinstance(args.number, list) else args.number
    tracker_ctx = make_tracker(
        args.enable_progress_tracker, work_dir=output_path, pipeline='perf', total_count=total_count
    )

    # Start benchmark
    with tracker_ctx:
        results = run_multi_benchmark(args, output_path=output_path)

    # Generate HTML report

    try:
        report_path = gen_perf_html_report(output_path, results, args)
        if report_path:
            logger.info(f'HTML report generated: {report_path}')
    except Exception as e:
        logger.warning(f'Failed to generate HTML report: {e}')

    return results


if __name__ == '__main__':
    args = Arguments.from_args(parse_args())
    result = run_perf_benchmark(args)
    print(result)
