# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
EvalScope: pip install "mteb<2"

2. Run eval task
"""
import os
import torch

from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()


def run_eval():

    # Prepare the config

    one_stage_task_cfg = {  # noqa
        'work_dir': 'outputs',
        'eval_backend': 'RAGEval',
        'eval_config': {
            'tool': 'MTEB',
            'model': [
                {
                    'model_name_or_path': 'AI-ModelScope/bge-large-zh',
                    'pooling_mode': 'cls',  # if not set, load from model config; use `cls` for bge series model
                    'max_seq_length': 512,
                    'prompt': '为这个句子生成表示以用于检索相关文章：',
                    'encode_kwargs': {
                        'batch_size': 512,
                    },
                }
            ],
            'eval': {
                'tasks': [
                    # 'TNews',
                    # 'CLSClusteringS2S',
                    # 'T2Reranking',
                    # 'ATEC',
                    # 'T2Retrieval',
                    # 'MMarcoRetrieval',
                    # 'DuRetrieval',
                    # 'CovidRetrieval',
                    # 'CmedqaRetrieval',
                    # 'EcomRetrieval',
                    'MedicalRetrieval',
                    # 'VideoRetrieval'
                ],
                'verbosity': 2,
                'overwrite_results': True,
                'top_k': 10,
                # 'limits': 1000,  # don't limit for retrieval task
            },
        },
    }

    two_stage_task_cfg = {
        'work_dir': 'outputs',
        'eval_backend': 'RAGEval',
        'eval_config': {
            'tool': 'MTEB',
            'model': [
                {
                    'model_name_or_path': 'AI-ModelScope/m3e-base',
                    'is_cross_encoder': False,
                    'max_seq_length': 512,
                    'prompt': '',
                    'model_kwargs': {'torch_dtype': torch.float16},
                    'encode_kwargs': {
                        'batch_size': 64,
                    },
                },
                {
                    'model_name_or_path': 'OpenBMB/MiniCPM-Reranker',
                    'is_cross_encoder': True,
                    'max_seq_length': 512,
                    'prompt': '为这个问题生成一个检索用的表示',
                    'model_kwargs': {'torch_dtype': torch.float16},
                    'encode_kwargs': {
                        'batch_size': 32,
                    },
                },
            ],
            'eval': {
                'tasks': ['T2Retrieval'],
                'verbosity': 2,
                'overwrite_results': True,
                'limits': 100,
            },
        },
    }

    from evalscope import TaskConfig
    api_task_cfg = TaskConfig(
        eval_backend='RAGEval',
        eval_config={
            'tool': 'MTEB',
            'model': [
                {
                    'model_name': 'text-embedding-v3',
                    'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                    'api_key': 'xxx',
                    'dimensions': 1024,
                    'encode_kwargs': {
                        'batch_size': 10,
                    },
                }
            ],
            'eval': {
                'tasks': [
                    'T2Retrieval',
                ],
                'verbosity': 2,
                'overwrite_results': True,
                'limits': 30,
            },
        },
    )

    # Run task
    # run_task(task_cfg=one_stage_task_cfg)
    run_task(task_cfg=two_stage_task_cfg)
    # run_eval_task(task_cfg=api_task_cfg)


if __name__ == '__main__':
    run_eval()
