# Copyright (c) Alibaba, Inc. and its affiliates.
"""CLI command for starting the EvalScope service (alias for 'service').

The 'app' command is deprecated.  Use 'evalscope service' instead.
"""
import warnings
from argparse import ArgumentParser

from evalscope.cli.base import CLICommand


def subparser_func(args):
    """Function which will be called for a specific sub parser."""
    return StartAppCMD(args)


class StartAppCMD(CLICommand):
    name = 'app'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """Define args for app command (deprecated alias for service)."""
        parser = parsers.add_parser(
            StartAppCMD.name,
            help='[DEPRECATED] Use "evalscope service" instead',
        )
        parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to (default: 0.0.0.0)')
        parser.add_argument('--port', type=int, default=9000, help='Port to listen on (default: 9000)')
        parser.add_argument(
            '--outputs', type=str, default=None, help='Root directory for evaluation outputs (default: ./outputs)'
        )
        parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
        parser.set_defaults(func=subparser_func)

    def execute(self):
        warnings.warn(
            'The "evalscope app" command is deprecated and will be removed in a future release. '
            'Use "evalscope service" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from evalscope.service import run_service

        run_service(host=self.args.host, port=self.args.port, debug=self.args.debug, outputs=self.args.outputs)
