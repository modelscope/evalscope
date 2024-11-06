import asyncio
import platform

from evalscope.perf.arguments import Arguments, parse_args
from evalscope.perf.benchmark import benchmark
from evalscope.perf.utils.signal_handler import add_signal_handlers


def run_perf_benchmark(args):
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop()
    if platform.system() != 'Windows':
        add_signal_handlers(loop)
    loop.run_until_complete(benchmark(args))


if __name__ == '__main__':
    args = Arguments.from_args(parse_args())
    run_perf_benchmark(args)
