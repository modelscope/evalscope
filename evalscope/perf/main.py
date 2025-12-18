import asyncio
import copy
import os
import platform
import threading
import time
from argparse import Namespace

from evalscope.utils.logger import configure_logging, get_logger
from evalscope.utils.model_utils import seed_everything
from .arguments import Arguments, parse_args
from .benchmark import benchmark
from .sla.sla_run import run_sla_auto_tune
from .utils.db_util import get_output_path
from .utils.handler import add_signal_handlers
from .utils.local_server import start_app
from .utils.log_utils import init_visualizer
from .utils.rich_display import print_summary

logger = get_logger()


def run_one_benchmark(args: Arguments, output_path: str = None):
    """Run a single benchmark with given parallel and number settings."""
    if isinstance(args.parallel, list):
        args.parallel = args.parallel[0]
    if isinstance(args.number, list):
        args.number = args.number[0]

    logger.info('Starting benchmark with args: ')
    logger.info(args)

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    # Only add signal handlers in main thread
    if platform.system() != 'Windows' and threading.current_thread() is threading.main_thread():
        try:
            add_signal_handlers(loop)
        except ValueError as e:
            logger.warning(f'Cannot add signal handlers (running in non-main thread): {e}')

    with args.output_context(output_path):
        metrics_result, percentile_result = loop.run_until_complete(benchmark(args))

    # Return unified format
    key = f'parallel_{args.parallel}_number_{args.number}'
    return {key: {'metrics': metrics_result, 'percentiles': percentile_result}}


def run_multi_benchmark(args: Arguments, output_path: str = None):
    """Run multiple benchmarks with different parallel and number combinations."""
    results = {}
    number_list = copy.deepcopy(args.number)
    parallel_list = copy.deepcopy(args.parallel)

    for i, (number, parallel) in enumerate(zip(number_list, parallel_list)):
        args.number = number
        args.parallel = parallel

        # Set up output path for each run
        cur_output_path = os.path.join(output_path, f'parallel_{parallel}_number_{number}')
        os.makedirs(cur_output_path, exist_ok=True)

        # Start the benchmark
        benchmark_result = run_one_benchmark(args, output_path=cur_output_path)
        results.update(benchmark_result)

        # Sleep between runs to avoid overwhelming the server
        if i < len(number_list) - 1:
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

    # Start benchmark
    if len(args.number) == 1:
        return run_one_benchmark(args, output_path=output_path)
    else:
        return run_multi_benchmark(args, output_path=output_path)


if __name__ == '__main__':
    args = Arguments.from_args(parse_args())
    result = run_perf_benchmark(args)
    print(result)
