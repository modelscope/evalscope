# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
from llmuses.cli.start_server import PerfCMD
def run_cmd():
    parser = argparse.ArgumentParser(
        'LLMUses Command Line tool', usage='llmuses <command> [<args>]')
    subparsers = parser.add_subparsers(help='llmuses commands helpers')
    
    PerfCMD.define_args(subparsers)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()