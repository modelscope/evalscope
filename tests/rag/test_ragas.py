# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dotenv import dotenv_values

env = dotenv_values('.env')
import unittest

from evalscope import TaskConfig, run_task
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
                    'generator_llm': {
                        'model_name': 'qwen-plus',  # 自定义聊天模型名称
                        'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  # 自定义基础URL
                        'api_key': env.get('DASHSCOPE_API_KEY', 'EMPTY'),  # 自定义API密钥
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
                        'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',
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
        from evalscope.backend.rag_eval.ragas.arguments import EvaluationArguments
        task_cfg = TaskConfig(
            eval_backend='RAGEval',
            eval_config=dict(
                tool='RAGAS',
                eval=EvaluationArguments(
                    testset_file='outputs/testset_chinese_with_answer_small.json',
                    critic_llm={
                        'model_name': 'qwen-plus',  # 自定义聊天模型名称
                        'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  # 自定义基础URL
                        'api_key': env.get('DASHSCOPE_API_KEY', 'EMPTY'),  # 自定义API密钥
                    },
                    embeddings={
                        'model_name': 'text-embedding-v1',
                        'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                        'api_key': env.get('DASHSCOPE_API_KEY', 'EMPTY'),
                        'dimensions': 1024,
                        'encode_kwargs': {
                            'batch_size': 10,
                        },
                    },
                    metrics=[
                        'Faithfulness',
                        'AnswerRelevancy',
                        'ContextPrecision',
                        'AnswerCorrectness',
                        # 'MultiModalFaithfulness',
                        # 'MultiModalRelevance',
                    ],
                ),
            ),
        )

        logger.info(f'>> Start to run task: {task_cfg}')

        run_task(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
