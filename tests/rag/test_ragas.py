# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import unittest
from evalscope.utils import test_level_list, is_module_installed
from evalscope.utils.logger import get_logger
from evalscope.run import run_task

logger = get_logger()


class TestRAGAS(unittest.TestCase):

    def setUp(self) -> None:
        self._check_env("ragas")

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f"{module_name} is installed.")
        else:
            raise ModuleNotFoundError(f"run: pip install {module_name}")

    @unittest.skipUnless(0 in test_level_list(), "skip test in current test level")
    def test_run_generate_dataset(self):
        task_cfg = {
            "eval_backend": "RAGEval",
            "eval_config": {
                "tool": "RAGAS",
                "testset_generation": {
                    "docs": ["README.md"],
                    "test_size": 5,
                    "output_file": "outputs/testset.json",
                    "distribution": {
                        "simple": 0.5,
                        "multi_context": 0.4,
                        "reasoning": 0.1,
                    },
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
            },
        }

        logger.info(f">> Start to run task: {task_cfg}")

        run_task(task_cfg)

    @unittest.skipUnless(0 in test_level_list(), "skip test in current test level")
    def test_run_rag_eval(self):
        task_cfg = {
            "eval_backend": "RAGEval",
            "eval_config": {
                "tool": "RAGAS",
                "eval": {
                    "testset_file": "outputs/testset.json",
                    "critic_llm": {
                        "model_name_or_path": "qwen/Qwen2-7B-Instruct",
                        "template_type": "qwen",
                    },
                    "embeddings": {
                        "model_name_or_path": "AI-ModelScope/m3e-base",
                    },
                    "metrics": [
                        "faithfulness",
                        "answer_relevancy",
                        "context_precision",
                        "answer_correctness",
                    ],
                },
            },
        }

        logger.info(f">> Start to run task: {task_cfg}")

        run_task(task_cfg)


if __name__ == "__main__":
    unittest.main(buffer=False)
