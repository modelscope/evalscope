# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.cli.base import ArgumentParserWithSubParsers, CLICommand


def subparser_func(args):
    """Function which will be called for a specific sub parser."""
    return EvalCMD(args)


class EvalCMD(CLICommand):
    name = 'eval'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParserWithSubParsers) -> None:
        """define args for create pipeline template command."""
        from evalscope.arguments import add_argument

        parser = parsers.add_parser(EvalCMD.name)
        add_argument(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        from evalscope.arguments import check_eval_type
        from evalscope.plugins import PluginError, load_plugins
        from evalscope.run import run_task

        try:
            load_plugins(self.args.plugins)
            check_eval_type(self.args.eval_type)
        except (PluginError, ValueError) as e:
            raise SystemExit(f'evalscope eval: error: {e}') from None
        run_task(self.args)
