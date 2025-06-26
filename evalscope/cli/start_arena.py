"""
Arena battle command for model comparison
"""

from argparse import ArgumentParser

from evalscope.cli.base import CLICommand
from evalscope.utils.logger import get_logger

logger = get_logger()


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return ArenaCMD(args)


class ArenaCMD(CLICommand):
    """
    Command for arena battle mode.
    """
    name = 'arena'

    @staticmethod
    def define_args(parsers: ArgumentParser):
        from evalscope.arena.arguments import add_argument

        parser = parsers.add_parser(ArenaCMD.name)
        add_argument(parser)
        parser.set_defaults(func=subparser_func)

    def __init__(self, args):
        self.args = args

    def execute(self):
        """
        Execute the arena battle command.
        """
        from evalscope.arena.arena_utils import run_arena_battle

        run_arena_battle(self.args)
