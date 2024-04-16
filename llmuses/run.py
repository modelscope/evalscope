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
from llmuses.constants import OutputsStructure
from llmuses.tools.combine_reports import ReportsRecorder
import os

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
    parser.add_argument('--qwen-path',
                        help='The qwen model id on modelscope, or local model dir.',
                        type=str,
                        required=False,
                        default="")
    parser.add_argument('--generation-config',
                        type=str,
                        help='The generation config, should be a string.',
                        required=False,
                        default='do_sample=False,repetition_penalty=1.0,max_new_tokens=512',
                        )
    parser.add_argument('--template-type',
                        type=str,
                        help='The template type for generation, should be a string.'
                             'Refer to `https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md` for more details.',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--datasets',
                        help='Dataset id list, align to the module name in llmuses.benchmarks',
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument('--dataset-args',
                        type=json.loads,
                        help='The dataset args, should be a json list string. The key of dict should be aligned to datasets,'
                             'e.g. {"humaneval": {"local_path": ["/to/your/path"]}}',
                        required=False,
                        default='{}')
    parser.add_argument('--dataset-dir',
                        help='The datasets dir. Use to specify the local datasets or datasets cache dir.'
                             'See --dataset-hub for more details.',
                        required=False,
                        default=DEFAULT_ROOT_CACHE_DIR)
    parser.add_argument('--dataset-hub',
                        help='The datasets hub, can be `ModelScope` or `HuggingFace` or `Local`. '
                             'Default to `ModelScope`.'
                             'If `Local`, the --dataset-dir should be local input data dir.'
                             'Otherwise, the --dataset-dir should be the cache dir for datasets.',
                        required=False,
                        default='ModelScope')
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
    parser.add_argument('--oss-args',
                        type=json.loads,
                        help='The oss args, should be a json string. The key of dict should covers "key_id, key_secret, oss_url, endpoint",'
                             'e.g. {"key_id": "XXX", "key_secret": "XXX", "endpoint": "https://oss-cn-hangzhou.aliyuncs.com", "oss_url": "oss://xxx/xxx/"}',
                        required=False,
                        default='{}')

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

def get_report_path(evaluator):
    report_dir: str = evaluator.outputs_structure[OutputsStructure.REPORTS_DIR]
    report_file_name: str = evaluator.dataset_name_or_path.replace('/', '_') + '.json'
    report_path: str = os.path.join(report_dir, report_file_name)
    return report_path

def set_oss_environ(args):
    os.environ['OSS_ACCESS_KEY_ID'] = args.get("key_id", "")
    os.environ['OSS_ACCESS_KEY_SECRET'] = args.get("key_secret", "")

def main():
    args = parse_args()
    logger.info(args)

    model_args = parse_str_args(args.model_args)
    generation_args = parse_str_args(args.generation_config)
    set_oss_environ(args.oss_args)

    # Parse args
    model_precision = model_args.get('precision', 'torch.float16')

    # Dataset hub
    dataset_hub: str = args.dataset_hub

    # Get model args
    if args.dry_run:
        from llmuses.models.dummy_chat_model import DummyChatModel
        model_id: str = 'dummy'
        qwen_model_id = 'qwen_dummy'
        model_revision: str = 'v1.0.0'
    else:
        model_id: str = args.model
        model_revision: str = model_args.get('revision', None)
        if model_revision == 'None':
            model_revision = eval(model_revision)
        qwen_model_id: str = args.qwen_path

    reports_recorder = ReportsRecorder(
        oss_url=args.oss_args.get("oss_url", ""),
        endpoint=args.oss_args.get("endpoint", "")
    )
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

        if dataset_hub == 'Local':
            dataset_path_list = args.dataset_args.get(dataset_name, {}).get('local_path')
            if not dataset_path_list:
                dataset_path_list = [dataset_name]
        elif dataset_hub == 'ModelScope':
            dataset_path_list = [imported_modules['DATASET_ID']]
        elif dataset_hub == 'HuggingFace':
            raise NotImplementedError('HuggingFace dataset hub is not supported yet.')
        else:
            raise ValueError(f'Unknown dataset hub: {dataset_hub}')

        print(f'> dataset_local_path_list: {dataset_path_list}')      # TODO: ONLY FOR TEST

        for dataset_name_or_path in dataset_path_list:
            if args.dry_run:
                from llmuses.models.dummy_chat_model import DummyChatModel
                model_adapter = DummyChatModel(model_cfg=dict())
                qwen_model_adapter = None
            else:
                # Init model adapter
                model_adapter = imported_modules['ModelAdapterClass'](model_id=model_id,
                                                                      model_revision=model_revision,
                                                                      device_map=model_args.get('device_map', 'auto'),
                                                                      torch_dtype=model_precision,
                                                                      cache_dir=args.work_dir,
                                                                      template_type=args.template_type,)
                qwen_model_adapter = imported_modules['ModelAdapterClass'](model_id=qwen_model_id,
                                                                           model_revision=None,
                                                                           device_map=model_args.get('device_map', 'auto'),
                                                                           torch_dtype=model_precision,
                                                                           cache_dir=args.work_dir,
                                                                           template_type='qwen',
                                                                           ) if len(qwen_model_id) > 0 else None

            if dataset_name == 'humaneval':
                problem_file: str = dataset_name_or_path

                evaluator = HumanevalEvaluator(problem_file=problem_file,
                                               model_id=model_id,
                                               model_revision=model_revision,
                                               model_adapter=model_adapter,
                                               outputs_dir=args.outputs,
                                               is_custom_outputs_dir=False,)
            else:
                evaluator = Evaluator(dataset_name_or_path=dataset_name_or_path,
                                      subset_list=imported_modules['SUBSET_LIST'],
                                      data_adapter=data_adapter,
                                      model_adapter=model_adapter,
                                      qwen_model_adapter=qwen_model_adapter,
                                      use_cache=args.mem_cache,
                                      root_cache_dir=args.work_dir,
                                      outputs_dir=args.outputs,
                                      is_custom_outputs_dir=False,
                                      datasets_dir=args.dataset_dir,
                                      datasets_hub=args.dataset_hub,
                                      stage=args.stage, )

            infer_cfg = generation_args or {}
            infer_cfg.update(dict(limit=args.limit))
            evaluator.eval(infer_cfg=infer_cfg, debug=args.debug)

            report_path = get_report_path(evaluator)
            reports_recorder.append_path(report_path, dataset_name)

    reports_recorder.dump_reports("./")


if __name__ == '__main__':
    # Usage: python3 llmuses/run.py --model ZhipuAI/chatglm2-6b --datasets arc --limit 10 --dry-run
    main()
    