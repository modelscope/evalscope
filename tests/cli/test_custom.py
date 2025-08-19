# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

from tests.utils import test_level_list

env = dotenv_values('.env')

import os
import subprocess
import unittest

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.run import run_task
from evalscope.utils.import_utils import is_module_installed
from evalscope.utils.logger import get_logger

os.environ['EVALSCOPE_LOG_LEVEL'] = 'DEBUG'

logger = get_logger()


class TestRunCustom(unittest.TestCase):
    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_custom_task(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='Qwen/Qwen3-0.6B',
            datasets=[
                'general_mcq',
                'general_qa'
            ],
            dataset_args={
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ],
                    'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'  # 问题模板
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ]
                }
            },
        )
        res = run_task(task_cfg=task_cfg)
        print(res)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_local_dataset(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen-plus',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= env.get('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                # 'mmlu',
                # 'race',
                'trivia_qa',
                # 'cmmlu',
                # 'humaneval',
                # 'gsm8k',
                # 'bbh',
                # 'competition_math',
                # 'arc',
                # 'ceval',
            ],
            dataset_args={
                'mmlu': {
                    'subset_list': ['elementary_mathematics', 'high_school_european_history', 'nutrition'],
                    'few_shot_num': 0,
                    'dataset_id': 'data/data/mmlu',
                },
                'ceval': {
                    'subset_list': [
                        'computer_network', 'operating_system', 'computer_architecture'
                    ],
                    'few_shot_num': 0,
                    'dataset_id': 'data/data/ceval',
                },
                'cmmlu': {
                    'subset_list': ['elementary_chinese'],
                    'dataset_id': 'data/data/cmmlu',
                    'few_shot_num': 0
                },
                'bbh': {
                    'subset_list': ['word_sorting', 'movie_recommendation'],
                },
                'humaneval': {
                    'metric_list': ['Pass@1', 'Pass@2', 'Pass@5'],
                },
                'trivia_qa': {
                    'dataset_id': 'data/data/trivia_qa',
                },
            },
            eval_batch_size=10,
            limit=5,
            debug=True,
            stream=True,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            ignore_errors=False,
        )

        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_general_no_answer(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen2.5-7b-instruct',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= env.get('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                'general_qa',
            ],
            dataset_args={
                'general_qa': {
                    'dataset_id': 'custom_eval/text/qa',
                    'subset_list': [
                        'arena',
                        # 'example'
                    ],
                }
            },
            eval_batch_size=10,
            limit=10,
            debug=True,
            stream=True,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            ignore_errors=False,
            judge_model_args={
                'model_id': 'qwen2.5-7b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096
                },
                'score_type': 'numeric',
                'prompt_template': """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 0 (worst) to 100 (best) by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\"

[Question]
{question}

[Response]
{pred}
"""
            },
            judge_worker_num=5,
            judge_strategy=JudgeStrategy.LLM,
        )

        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_general_with_answer(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen-plus',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= env.get('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                'general_qa',
            ],
            dataset_args={
                'general_qa': {
                    'dataset_id': 'custom_eval/text/qa',
                    'subset_list': [
                        'example'
                    ],
                }
            },
            eval_batch_size=10,
            limit=10,
            debug=True,
            stream=True,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            ignore_errors=False,
            judge_model_args={
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096
                },
                'score_type': 'pattern',
            },
            judge_worker_num=1,
            judge_strategy=JudgeStrategy.LLM_RECALL,
            use_cache='outputs/20250818_170420'
        )

        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_general_arena(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model_id='Arena',
            datasets=[
                'general_arena',
            ],
            dataset_args={
                'general_arena': {
                    'extra_params':{
                        'models':[
                            {
                                'name': 'qwen2.5-7b',
                                'report_path': 'outputs/20250819_165034/reports/qwen2.5-7b-instruct'
                            },
                            {
                                'name': 'qwen2.5-72b',
                                'report_path': 'outputs/20250819_164926/reports/qwen2.5-72b-instruct'
                            }
                        ],
                        'baseline': 'qwen2.5-72b'
                    }
                }
            },
            eval_batch_size=10,
            limit=10,
            debug=True,
            stream=True,
            ignore_errors=False,
            judge_model_args={
                'model_id': 'qwen-plus',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 8000
                },
            },
            judge_worker_num=5,
            # use_cache='outputs/20250819_173546'
        )

        run_task(task_cfg=task_cfg)
