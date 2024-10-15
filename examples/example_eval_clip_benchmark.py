# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
EvalScope: pip install webdataset

2. Run eval task
"""
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
import torch

logger = get_logger()


def run_eval():

    # Prepare the config

    task_cfg = {
        "eval_backend": "RAGEval",
        "eval_config": {
            "tool": "clip_benchmark",
            "eval": {
                "models": [
                    # {
                    #     "model_name": "AI-ModelScope/chinese-clip-vit-large-patch14-336px",
                    # }
                    {
                        "model_name": "internvl2-8b",
                        "api_base": "http://localhost:8000/v1",
                    }
                ],
                # "dataset_name": ["muge", "flickr8k"],
                "dataset_name": ["custom"],
                "data_dir": "custom_eval/multimodal/text-image-retrieval",
                "split": "test",
                "task": "image_caption",
                "batch_size": 128,
                "num_workers": 1,
                "verbose": True,
                "skip_existing": False,
                "limit": 1000,
            },
        },
    }

    # Run task
    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    run_eval()
