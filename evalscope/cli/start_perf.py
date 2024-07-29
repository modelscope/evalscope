# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import abstractmethod
import os, sys, time
from argparse import ArgumentParser
import subprocess


from evalscope.cli.base import CLICommand
from evalscope.perf.http_client import add_argument, run_perf_benchmark

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
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
        
        
        
        