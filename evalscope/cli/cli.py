# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
from evalscope.cli.start_perf import PerfBenchCMD


def run_cmd():
    parser = argparse.ArgumentParser(
        'EvalScope Command Line tool', usage='evalscope <command> [<args>]')
    subparsers = parser.add_subparsers(help='Performance benchmark command line.')
    
    PerfBenchCMD.define_args(subparsers)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    cmd = args.func(args)
    cmd.execute()
    # --url 'http://11.122.132.12:8000/v1/chat/completions' --parallel 1 --model 'qwen' --dataset 'datasets/LongAlpaca-12k.jsonl'  --log-every-n-query 1 --read-timeout=120  --parser 'openai.longalpaca_12k_qwen.py' -n 10 --max-prompt-length 128000 --tokenizer-path ''


if __name__ == '__main__':
    run_cmd()
