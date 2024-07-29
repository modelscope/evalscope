# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa

import argparse
import torch

from evalscope.benchmarks.ceval import DATASET_ID as CEVAL_EXAM
from evalscope.benchmarks.mmlu import DATASET_ID as MMLU
from evalscope.benchmarks.hellaswag import DATASET_ID as HELLA_SWAG
from evalscope.benchmarks.arc import DATASET_ID as ARC
from evalscope.benchmarks.truthful_qa import DATASET_ID as TRUTHFUL_QA
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR
from evalscope.evaluator import Evaluator
from evalscope.models.model_adapter import MultiChoiceModelAdapter, ContinuationLogitsModelAdapter
from evalscope.utils.logger import get_logger

logger = get_logger()

# TODO: add more precision
MODEL_PRECISION_MAP = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}

"""
Run evaluation process for ModelScope Leaderboard.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on a model')

    parser.add_argument('--model', help='Model id from modelscope or huggingface.', required=True)
    parser.add_argument('--revision', help='Model revision.', required=False, default=None)
    parser.add_argument('--precision', help='Model precision.', default='bf16')
    parser.add_argument('--work-dir', help='root work cache dir.', default=None)
    parser.add_argument('--outputs-dir', help='Outputs dir.', default='outputs')
    parser.add_argument('--datasets-dir', help='Datasets dir.', default=DEFAULT_ROOT_CACHE_DIR)
    parser.add_argument('--device-map', help='device map.', default='auto')
    parser.add_argument('--max-eval-size', type=int, help='Max evaluation samples num for each subset', default=None)
    parser.add_argument('--dataset-id', help='Dataset id on modelscope', required=False, default=None)

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

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger.info(args)

    # Customize your target datasets here
    all_benchmarks = [CEVAL_EXAM, MMLU, ARC, HELLA_SWAG, TRUTHFUL_QA]

    dataset_id = args.dataset_id
    if dataset_id is None:
        datasets = all_benchmarks
    elif dataset_id in all_benchmarks:
        datasets = [dataset_id]
    else:
        raise ValueError(f'Unknown dataset: {dataset_id}, Supported datasets: {all_benchmarks}')

    # Get model instance
    if args.dry_run:
        from evalscope.models.dummy_chat_model import DummyChatModel
        model_adapter = DummyChatModel(model_cfg=dict())        # TODO
        model_id: str = 'dummy'
        model_revision: str = 'v1.0.0'
        model_precision = MODEL_PRECISION_MAP.get(args.precision, torch.bfloat16)
    else:
        model_id: str = args.model
        model_revision: str = args.revision
        model_precision = MODEL_PRECISION_MAP.get(args.precision, torch.bfloat16)

        model_adapter = MultiChoiceModelAdapter(model_id=model_id,
                                                device_map=args.device_map,
                                                torch_dtype=model_precision,
                                                model_revision=model_revision,)

    # Evaluate on each dataset
    for dataset_name in datasets:
        if dataset_name == CEVAL_EXAM:
            from evalscope.benchmarks.ceval import CEVALAdapter
            data_adapter = CEVALAdapter()
        elif dataset_name == MMLU:
            from evalscope.benchmarks.mmlu import MMLUAdapter
            data_adapter = MMLUAdapter()
        elif dataset_name == ARC:
            from evalscope.benchmarks.arc import ARCAdapter
            data_adapter = ARCAdapter()
        elif dataset_name == HELLA_SWAG:
            # Note: HellaSwag should run few-shot eval
            from evalscope.benchmarks.hellaswag import HellaSwagAdapter
            data_adapter = HellaSwagAdapter()
        elif dataset_name == TRUTHFUL_QA:
            from evalscope.benchmarks.truthful_qa import TruthfulQaAdapter
            data_adapter = TruthfulQaAdapter()

        # TODO: add more datasets here
        else:
            raise ValueError(f'Unknown dataset: {dataset_name}')

        # TODO: add mapping
        if dataset_name in {TRUTHFUL_QA, HELLA_SWAG} and not args.dry_run:
            model_adapter = ContinuationLogitsModelAdapter(model_id=model_id,
                                                           device_map=args.device_map,
                                                           torch_dtype=model_precision,
                                                           model_revision=model_revision, )

        root_work_dir = args.work_dir if args.work_dir is not None else DEFAULT_ROOT_CACHE_DIR
        evaluator = Evaluator(dataset_name_or_path=dataset_name,
                              subset_list=None,
                              data_adapter=data_adapter,
                              model_adapter=model_adapter,
                              use_cache=args.mem_cache,
                              root_cache_dir=root_work_dir,
                              outputs_dir=args.outputs_dir,
                              is_custom_outputs_dir=True,
                              datasets_dir=args.datasets_dir, )

        infer_cfg = dict(max_length=2048, limit=args.max_eval_size)
        evaluator.eval(infer_cfg=infer_cfg, debug=args.debug)


if __name__ == '__main__':
    main()

    # Usage:
    # python evalscope/run_ms.py --model ZhipuAI/chatglm2-6b --precision fp16 --dry-run --dataset-id modelscope/mmlu --limit 10

