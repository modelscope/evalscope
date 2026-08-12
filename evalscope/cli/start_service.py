# Copyright (c) Alibaba, Inc. and its affiliates.
"""CLI command for starting the EvalScope Flask service."""
import os
from argparse import ArgumentParser, ArgumentTypeError

from evalscope.cli.base import CLICommand


def subparser_func(args):
    """Function which will be called for a specific sub parser."""
    return ServiceCMD(args)


def existing_directory(value: str) -> str:
    """Return an existing directory path or produce an argparse error."""
    path = os.path.abspath(value)
    if not os.path.isdir(path):
        raise ArgumentTypeError(f'output directory does not exist: {path}')
    return path


class ServiceCMD(CLICommand):
    name = 'service'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """Define args for service command."""
        parser = parsers.add_parser(
            ServiceCMD.name, help='Start the EvalScope Flask service for eval and perf endpoints'
        )
        parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to (default: 0.0.0.0)')
        parser.add_argument('--port', type=int, default=9000, help='Port to listen on (default: 9000)')
        parser.add_argument(
            '--outputs',
            type=existing_directory,
            default=None,
            help='Root directory for evaluation outputs (default: ./outputs). '
            'The web dashboard will use this as the default scan path.'
        )
        parser.add_argument('--debug', action='store_true', default=False, help='Enable Flask debug mode')
        parser.set_defaults(func=subparser_func)

    def execute(self):
        """Execute the service command."""
        from evalscope.service import run_service

        run_service(host=self.args.host, port=self.args.port, debug=self.args.debug, outputs=self.args.outputs)
