# Copyright (c) Alibaba, Inc. and its affiliates.
from argparse import ArgumentParser

from evalscope.cli.base import CLICommand
from evalscope.utils.logger import get_logger

logger = get_logger()


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return PerfBenchCMD(args)


class PerfBenchCMD(CLICommand):
    name = 'perf'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """Define the performance benchmark command arguments."""
        from evalscope.perf.config.cli import add_cli_arguments

        parser = parsers.add_parser(PerfBenchCMD.name)
        add_cli_arguments(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        try:
            from evalscope.perf import PerfError, run_perf
            from evalscope.perf.config.cli import config_from_namespace
        except ImportError as e:
            raise ImportError(
                f'Failed to import the EvalScope perf API, due to {e}. '
                "Please run `pip install 'evalscope[perf]'`."
            )

        try:
            run_perf(config_from_namespace(self.args))
        except (PerfError, ValueError) as e:
            logger.error(str(e))
            raise SystemExit(2) from e
