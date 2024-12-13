import asyncio
import logging
import os
import platform
from argparse import Namespace

from evalscope.perf.arguments import Arguments, parse_args
from evalscope.perf.benchmark import benchmark
from evalscope.perf.utils.db_util import get_output_path
from evalscope.perf.utils.handler import add_signal_handlers
from evalscope.utils.logger import get_logger
from evalscope.utils.utils import seed_everything

logger = get_logger()


def run_perf_benchmark(args):
    if isinstance(args, dict):
        args = Arguments(**args)
    elif isinstance(args, Namespace):
        args = Arguments.from_args(args)
    seed_everything(args.seed)

    # Setup logger and output
    args.outputs_dir = get_output_path(args)
    get_logger(log_file=os.path.join(args.outputs_dir, 'benchmark.log'), force=True)

    if args.debug:
        get_logger(log_level=logging.DEBUG, force=True)

    logger.info('Starting benchmark...')
    logger.info(args)

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    if platform.system() != 'Windows':
        add_signal_handlers(loop)
    loop.run_until_complete(benchmark(args))


if __name__ == '__main__':
    args = Arguments.from_args(parse_args())
    run_perf_benchmark(args)
