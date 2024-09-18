# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
EvalScope: pip install mteb

2. Run eval task
"""
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()


def run_eval():
    task_cfg = {
        "eval_backend": "RAGEval",
        "eval_config": {
            "tool": "RAGAS",
            "testset_generation": {
                "docs": ["README.md"],
                "test_size": 10,
                "output_file": "outputs/testset.json",
                "distribution": {"simple": 0.5, "multi_context": 0.4, "reasoning": 0.1},
                # The generator_llm is the component that generates the questions, and evolves the question to make it more relevant.
                # The critic_llm is the component that filters the questions and nodes based on the question and node relevance.
                "generator_llm": {
                    "model_name_or_path": "qwen/Qwen2-7B-Instruct",
                    "template_type": "qwen",
                },
                "critic_llm": {
                    "model_name_or_path": "QwenCollection/Ragas-critic-llm-Qwen1.5-GPTQ",
                    "template_type": "qwen",
                },
                "embeddings": {
                    "model_name_or_path": "AI-ModelScope/m3e-base",
                },
            },
            "eval": {
                "testset_file": "outputs/testset.json",
                "critic_llm": {
                    "model_name_or_path": "qwen/Qwen2-7B-Instruct",
                    "template_type": "qwen",
                },
                "embeddings": {
                    "model_name_or_path": "Jerry0/m3e-base",
                },
            },
        },
    }

    # Run task
    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    run_eval()