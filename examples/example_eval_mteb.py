# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
EvalScope: pip install mteb

2. Run eval task
"""
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
import torch

logger = get_logger()


def run_eval():

    # Prepare the config

    one_stage_task_cfg = {
        "eval_backend": "RAGEval",
        "eval_config": {
            "tool": "MTEB",
            "model": [
                {
                    "model_name_or_path": "AI-ModelScope/m3e-base",
                    "pooling_mode": None,  # load from model config
                    "max_seq_length": 512,
                    "prompt": "",
                    "model_kwargs": {"torch_dtype": "auto"},
                    "encode_kwargs": {
                        "batch_size": 128,
                    },
                }
            ],
            "eval": {
                "tasks": [
                    "TNews",
                    "CLSClusteringS2S",
                    "T2Reranking",
                    "T2Retrieval",
                    "ATEC",
                ],
                "verbosity": 2,
                "output_folder": "outputs",
                "overwrite_results": True,
                "limits": 500,
            },
        },
    }

    two_stage_task_cfg = {
        "eval_backend": "RAGEval",
        "eval_config": {
            "tool": "MTEB",
            "model": [
                {
                    "model_name_or_path": "AI-ModelScope/m3e-base",
                    "is_cross_encoder": False,
                    "max_seq_length": 512,
                    "prompt": "",
                    "model_kwargs": {"torch_dtype": "auto"},
                    "encode_kwargs": {
                        "batch_size": 64,
                    },
                },
                {
                    "model_name_or_path": "OpenBMB/MiniCPM-Reranker",
                    "is_cross_encoder": True,
                    "max_seq_length": 512,
                    "prompt": "为这个问题生成一个检索用的表示",
                    "model_kwargs": {"torch_dtype": "auto"},
                    "encode_kwargs": {
                        "batch_size": 32,
                    },
                },
            ],
            "eval": {
                "tasks": ["T2Retrieval"],
                "verbosity": 2,
                "output_folder": "outputs",
                "overwrite_results": True,
                "limits": 100,
            },
        },
    }

    # Run task
    run_task(task_cfg=two_stage_task_cfg)


if __name__ == "__main__":
    run_eval()
