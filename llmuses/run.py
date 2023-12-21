# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa

import json
import argparse
import torch        # noqa

from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.evaluator import Evaluator
from llmuses.evaluator.evaluator import HumanevalEvaluator
from llmuses.utils import import_module_util
from llmuses.utils.logger import get_logger

logger = get_logger()

"""
Run evaluation for LLMs.
"""

BENCHMARK_PATH_PREFIX = 'llmuses.benchmarks.'
MEMBERS_TO_IMPORT = ['DATASET_ID', 'SUBSET_LIST', 'DataAdapterClass', 'ModelAdapterClass']


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on benchmarks for LLMs.')

    parser.add_argument('--model',
                        help='The model id on modelscope, or local model dir.',
                        type=str,
                        required=True)
    parser.add_argument('--model-args',
                        type=str,
                        help='The model args, should be a string.',
                        required=False,
                        default='revision=None,precision=torch.float16,device_map=auto'
                        )
    parser.add_argument('--generation-config',
                        type=str,
                        help='The generation config, should be a string.',
                        required=False,
                        default='do_sample=False,repetition_penalty=1.0,max_new_tokens=512',
                        )
    parser.add_argument('--datasets',
                        help='Dataset id list, align to the module name in llmuses.benchmarks',
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument('--dataset-args',
                        type=json.loads,
                        help='The dataset args, should be a json string. The key of dict should be aligned to datasets,'
                             'e.g. {"humaneval": {"local_path": "/to/your/path"}}',
                        required=False,
                        default='{}')
    parser.add_argument('--outputs',
                        help='Outputs dir.',
                        default='outputs')
    parser.add_argument('--work-dir',
                        help='The root cache dir.',
                        required=False,
                        default=DEFAULT_ROOT_CACHE_DIR)
    parser.add_argument('--limit',
                        type=int,
                        help='Max evaluation samples num for each subset. Default to None, which means no limit.',
                        default=None)
    parser.add_argument('--debug',
                        help='Debug mode, will print information for debugging.',
                        action='store_true',
                        default=False)
    parser.add_argument('--dry-run',
                        help='Dry run in single processing mode.',
                        action='store_true',
                        default=False)
    parser.add_argument('--mem-cache',
                        help='To use memory cache or not.',
                        action='store_true',
                        default=False)
    parser.add_argument('--stage',      # TODO
                        help='The stage of evaluation pipeline, '
                             'can be `all`, `infer`, `review`, `report`. Default to `all`.',
                        type=str,
                        default='all')

    args = parser.parse_args()

    return args


def parse_str_args(str_args: str):
    assert isinstance(str_args, str), 'args should be a string.'
    arg_list: list = str_args.strip().split(',')
    arg_list = [arg.strip() for arg in arg_list]
    arg_dict: dict = dict([arg.split('=') for arg in arg_list])

    final_args = dict()
    for k, v in arg_dict.items():
        try:
            final_args[k] = eval(v)
        except:
            final_args[k] = v

    return final_args


def main():
    args = parse_args()
    logger.info(args)

    model_args = parse_str_args(args.model_args)
    generation_args = parse_str_args(args.generation_config)

    # Parse args
    model_precision = model_args.get('precision', 'torch.float16')

    # Get model args
    if args.dry_run:
        from llmuses.models.dummy_chat_model import DummyChatModel
        model_id: str = 'dummy'
        model_revision: str = 'v1.0.0'
    else:
        model_id: str = args.model
        model_revision: str = model_args.get('revision', None)
        if model_revision == 'None':
            model_revision = eval(model_revision)

    datasets_list = args.datasets
    for dataset_name in datasets_list:
        # Get imported_modules
        imported_modules = import_module_util(BENCHMARK_PATH_PREFIX, dataset_name, MEMBERS_TO_IMPORT)
        # Init data adapter
        data_adapter = imported_modules['DataAdapterClass']()

        if dataset_name == 'humaneval' and args.dataset_args.get('humaneval', {}).get('local_path') is None:
            raise ValueError('Please specify the local problem path of humaneval dataset in --dataset-args,'
                             'e.g. {"humaneval": {"local_path": "/to/your/path"}}, '
                             'And refer to https://github.com/openai/human-eval/tree/master#installation to install it,'
                             'Note that you need to enable the execution code in the human_eval/execution.py first.')

        if args.dry_run:
            from llmuses.models.dummy_chat_model import DummyChatModel
            model_adapter = DummyChatModel(model_cfg=dict())
        else:
            # Init model adapter
            model_adapter = imported_modules['ModelAdapterClass'](model_id=model_id,
                                                                  model_revision=model_revision,
                                                                  device_map=model_args.get('device_map', 'auto'),
                                                                  torch_dtype=model_precision,)

        if dataset_name == 'humaneval':
            problem_file: str = args.dataset_args.get('humaneval', {}).get('local_path')

            evaluator = HumanevalEvaluator(problem_file=problem_file,
                                           model_id=model_id,
                                           model_revision=model_revision,
                                           model_adapter=model_adapter,
                                           outputs_dir=args.outputs,)
        else:
            evaluator = Evaluator(dataset_name_or_path=imported_modules['DATASET_ID'],
                                  subset_list=imported_modules['SUBSET_LIST'],
                                  data_adapter=data_adapter,
                                  model_adapter=model_adapter,
                                  use_cache=args.mem_cache,
                                  root_cache_dir=args.work_dir,
                                  outputs_dir=args.outputs,
                                  datasets_dir=args.work_dir,
                                  stage=args.stage, )

        infer_cfg = generation_args or {}
        infer_cfg.update(dict(limit=args.limit))
        evaluator.eval(infer_cfg=infer_cfg, debug=args.debug)


if __name__ == '__main__':
    # Usage: python llmuses/run.py --model ZhipuAI/chatglm2-6b --datasets mmlu hellaswag --limit 10
    main()
