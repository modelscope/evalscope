# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse

from evalscope.cli.start_app import StartAppCMD
from evalscope.cli.start_eval import EvalCMD
from evalscope.cli.start_perf import PerfBenchCMD


def run_cmd():
    parser = argparse.ArgumentParser('EvalScope Command Line tool', usage='evalscope <command> [<args>]')
    subparsers = parser.add_subparsers(help='EvalScope command line helper.')

    PerfBenchCMD.define_args(subparsers)
    EvalCMD.define_args(subparsers)
    StartAppCMD.define_args(subparsers)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()
