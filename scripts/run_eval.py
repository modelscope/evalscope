# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '..'))

import argparse
from evals.utils.utils import jsonl_to_list
from evals.preprocess.tokenizers.gpt2_tokenizer import GPT2Tokenizer, get_tokenized_string

from evals.tasks.task_eval_qwen_code import TaskQwenCodeEval
from evals.tasks.task_eval_qwen_generation import TaskQwenGenerationEval
from evals.tasks.task_eval_qwen_math import TaskQwenMathEval


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate on generation results.')
    parser.add_argument('--input', type=str, required=True, help='Evaluation file name.')
    parser.add_argument('--task_cfg', type=str, required=True, help='Evaluation task config file.')
    parser.add_argument('--eval-type', type=str, required=True, choices=['math', 'code', 'rouge', 'rm'],
                        help='Evaluation file type. (will be refactored)')
    parser.add_argument('--pass@k', type=int, required=False, default=4,
                        help='k in pass@k')
    parser.add_argument('--output', nargs='?', type=str, default=None, help='Output JSON file name.')
    parser.add_argument('--select-tags', choices=[None, 'task', 'ability', 'industry', 'level'], default=None,
                        help='eval sample with dimension of tag')
    parser.add_argument(
        '--vocab-file', type=str, default='evals/metrics/resources/gpt2-zhcn3-v4.json',
        help='Vocab file name',
    )
    parser.add_argument(
        '--merge-file', type=str, default='evals/metrics/resources/gpt2-zhcn3-v4.bpe',
        help='Merge file name',
    )

    args = parser.parse_args()
    return args


def preprocess(args):
    tokenizer = GPT2Tokenizer(
        args.vocab_file, args.merge_file, errors='replace',
        special_tokens=[], max_len=None,
    )
    assert args.input and os.path.exists(args.input)
    data_l = jsonl_to_list(args.input)
    for data in data_l:
        targets = data.get('target', [])
        if isinstance(targets, str):
            targets = [targets]
        tok, tok_str = get_tokenized_string(tokenizer, targets)
        data['reference_tok'] = tok
        data['reference_tok_str'] = tok_str

        gens = data.get('gen', [])
        if isinstance(gens, str):
            gens = [gens]
        tok, tok_str = get_tokenized_string(tokenizer, gens)
        data['gen_tok'] = tok
        data['gen_tok_str'] = tok_str
    return data_l


def run_eval(args):
    data_list = preprocess(args)

    if args.eval_type == 'math':
        task = TaskQwenMathEval(prompts=data_list, task_cfg=args.task_cfg)
    elif args.eval_type == 'code':
        task = TaskQwenCodeEval(prompts=data_list, task_cfg=args.task_cfg)
    elif args.eval_type == 'rouge':
        task = TaskQwenGenerationEval(prompts=data_list, task_cfg=args.task_cfg)
    else:
        raise ValueError(f'Unsupported eval type {args.eval_type}')

    task.run()


if __name__ == '__main__':
    args = get_args()
    run_eval(args)
