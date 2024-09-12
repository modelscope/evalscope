# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
EvalScope: pip install mteb

2. Run eval task
"""
from evalscope.run import run_task
from evalscope.summarizer import Summarizer
from evalscope.utils.logger import get_logger


logger = get_logger()

def run_eval():

    # Prepare the config
    # model_name = "Jerry0/m3e-base"
    # model_name = "../models/embedding/MiniCPM-Embedding"
    # model_name = "Xorbits/bge-reranker-base"
    model_name = "OpenBMB/MiniCPM-Reranker"
    
    # # Option 1: Use dict format
    task_cfg = {
        "eval_backend": "RAGEval",
        "eval_config": {
            "tool": "MTEB",
            "model_name_or_path": model_name,
            "pooling_mode": None,  # load from model config
            "max_seq_length": 512,
            "model_kwargs": {"torch_dtype": "auto"},
            "encode_kwargs": {
                "show_progress_bar": True,
                "batch_size": 32,
                "normalize_embeddings": True,
            },
            "tasks": ["T2Retrieval"],
            "instructions": {"T2Retrieval": "为这个问题生成一个检索用的表示"},
            "verbosity": 2,
            "output_folder": "outputs",
            "overwrite_results": True,
            "limits": 100,
            "hub": "modelscope",
        },
    }

    # Option 2: Use yaml file
    # task_cfg = "examples/tasks/eval_vlm_swift.yaml"

    # Run task
    run_task(task_cfg=task_cfg)

    # [Optional] Get the final report with summarizer
    # logger.info(">> Start to get the report with summarizer ...")
    # report_list = Summarizer.get_report_from_cfg(task_cfg)
    # logger.info(f"\n>> The report list: {report_list}")


if __name__ == "__main__":
    run_eval()
