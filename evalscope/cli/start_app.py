# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from argparse import ArgumentParser

from evalscope.cli.base import CLICommand
from evalscope.report.app import add_argument, create_app


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return StartAppCMD(args)


class StartAppCMD(CLICommand):
    name = 'app'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        parser = parsers.add_parser(StartAppCMD.name)
        add_argument(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        create_app(self.args)
