# Copyright (c) Alibaba, Inc. and its affiliates.
from argparse import ArgumentParser

from evalscope.cli.base import CLICommand


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return EvalCMD(args)


class EvalCMD(CLICommand):
    name = 'eval'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        from evalscope.arguments import add_argument

        parser = parsers.add_parser(EvalCMD.name)
        add_argument(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        from evalscope.run import run_task

        run_task(self.args)
