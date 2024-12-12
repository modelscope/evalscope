# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from evalscope.run import run_task
from evalscope.utils import is_module_installed, test_level_list
from evalscope.utils.logger import get_logger

logger = get_logger()


class TestRAGAS(unittest.TestCase):

    def setUp(self) -> None:
        self._check_env('ragas')

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_generate_dataset(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'RAGAS',
                'testset_generation': {
                    'docs': ['README_zh.md'],
                    'test_size': 5,
                    'output_file': 'outputs/testset.json',
                    'distribution': {
                        'simple': 0.5,
                        'multi_context': 0.4,
                        'reasoning': 0.1,
                    },
                    'generator_llm': {
                        'model_name_or_path': 'qwen/Qwen2-7B-Instruct',
                    },
                    'embeddings': {
                        'model_name_or_path': 'AI-ModelScope/m3e-base',
                    },
                    'language': 'chinese',
                },
            },
        }

        logger.info(f'>> Start to run task: {task_cfg}')

        run_task(task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_rag_eval(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'RAGAS',
                'eval': {
                    'testset_file': 'outputs/testset_chinese_with_answer.json',
                    'critic_llm': {
                        'model_name_or_path': 'qwen/Qwen2-7B-Instruct',
                    },
                    'embeddings': {
                        'model_name_or_path': 'AI-ModelScope/m3e-base',
                    },
                    'metrics': [
                        'Faithfulness',
                        'AnswerRelevancy',
                        'ContextPrecision',
                        'AnswerCorrectness',
                    ],
                },
            },
        }

        logger.info(f'>> Start to run task: {task_cfg}')

        run_task(task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_rag_eval_api(self):
        task_cfg = {
            'eval_backend': 'RAGEval',
            'eval_config': {
                'tool': 'RAGAS',
                'eval': {
                    'testset_file':
                    'outputs/testset.json',
                    'critic_llm': {
                        'model_name': 'gpt-4o-mini',  # 自定义聊天模型名称
                        'api_base': 'http://127.0.0.1:8088/v1',  # 自定义基础URL
                        'api_key': 'xxxx',  # 你的API密钥
                    },
                    'embeddings': {
                        'model_name_or_path': 'AI-ModelScope/m3e-base',
                    },
                    'metrics': [
                        'Faithfulness',
                        'AnswerRelevancy',
                        'ContextPrecision',
                        'AnswerCorrectness',
                        'MultiModalFaithfulness',
                        'MultiModalRelevance',
                    ],
                },
            },
        }

        logger.info(f'>> Start to run task: {task_cfg}')

        run_task(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
