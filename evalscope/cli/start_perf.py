# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from argparse import ArgumentParser

from evalscope.cli.base import CLICommand
from evalscope.perf.arguments import add_argument
from evalscope.perf.main import run_perf_benchmark


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
        """ define args for create pipeline template command.
        """
        parser = parsers.add_parser(PerfBenchCMD.name)
        add_argument(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        run_perf_benchmark(self.args)
