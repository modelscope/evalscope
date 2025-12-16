# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import os
import unittest

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

os.environ['EVALSCOPE_LOG_LEVEL'] = 'DEBUG'

logger = get_logger()

datasets=[
    'iquiz',
    'ifeval',
    'mmlu',
    'mmlu_pro',
    'musr',
    'process_bench',
    'race',
    # 'trivia_qa',
    'cmmlu',
    'humaneval',
    'gsm8k',
    'bbh',
    'competition_math',
    'math_500',
    'aime24',
    'gpqa_diamond',
    'arc',
    'ceval',
    'hellaswag',
    'general_mcq',
    'general_qa',
    'super_gpqa',
    # 'live_code_bench',
    'mmlu_redux',
    'simple_qa',
    'chinese_simpleqa',
    'alpaca_eval',
    'arena_hard',
    'maritime_bench',
    'drop',
    'winogrande',
    'tool_bench',
    'frames',
    'docmath',
    'needle_haystack',
    'bfcl_v3',
    'hle',
    'tau_bench',
]

# Reverse the datasets list to ensure the order is from most recent to oldest
datasets.reverse()

dataset_args={
    'mmlu': {
        'subset_list': ['elementary_mathematics', 'high_school_european_history', 'nutrition'],
        'few_shot_num': 0
    },
    'mmlu_pro': {
        'subset_list': ['math', 'health'],
        'few_shot_num': 4
    },
    'ceval': {
        'subset_list': [
            'computer_network', 'operating_system', 'computer_architecture'
        ],
        'few_shot_num': 0
    },
    'cmmlu': {
        'subset_list': ['elementary_chinese'],
        'few_shot_num': 0
    },
    'bbh': {
        'subset_list': ['word_sorting', 'movie_recommendation'],
    },
    'gpqa_diamond': {
        'few_shot_num': 0,
    },
    'competition_math': {
        'subset_list': ['Level 1']
    },
    'math_500': {
        'subset_list': ['Level 1']
    },
    'process_bench': {
        'subset_list': ['gsm8k'],
    },
    'musr': {
        'subset_list': ['murder_mysteries']
    },
    'general_mcq': {
        'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
        'subset_list': [
            'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
        ],
    },
    'general_qa': {
        'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
        'subset_list': [
            'example',  # 评测数据集名称，上述 *_dev.csv 中的 *
            # 'test'
        ]
    },
    'super_gpqa': {
        'subset_list': ['Philosophy', 'Education'],
        'few_shot_num': 0
    },
    'live_code_bench': {
        'subset_list': ['v4_v5'],
        'extra_params': {
            'start_date': '2024-12-01',
            'end_date': '2025-01-01'
        },
    },
    'chinese_simpleqa': {
        'subset_list': ['中华文化']
    },
    'mmlu_redux':{
        'subset_list': ['abstract_algebra']
    },
    'docmath':{
        'subset_list': ['simpshort_testmini']
    },
    'bfcl_v3':{
        'subset_list': ['simple', 'multiple']
    },
    'hle': {
        'subset_list': ['Math', 'Other'],
    },
    'tau_bench': {
        'extra_params': {
            'user_model': 'qwen-plus',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        },
        'subset_list': ['airline'],
    },
}

class TestRun(unittest.TestCase):
    def test_benchmarks(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen-plus',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= env.get('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=datasets,
            dataset_args=dataset_args,
            eval_batch_size=1,
            limit=1,
            stream=True,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            judge_worker_num=5,
            judge_strategy=JudgeStrategy.AUTO,
            judge_model_args={
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
            }
        )

        run_task(task_cfg=task_cfg)

    def test_vlm_benchmark(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen-vl-plus',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= env.get('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                'mmmu',
                'math_vista',
            ],
            dataset_args={
                'mmmu': {
                    'subset_list': ['Accounting']
                },
                'math_vista': {
                    'subset_list': ['default']
                }
            },
            eval_batch_size=1,
            limit=1,
            stream=True,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
                'image_height': 512,
                'image_width': 512,
                'image_num': 2,
            },
            judge_worker_num=5,
            judge_strategy=JudgeStrategy.AUTO,
            judge_model_args={
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
            }
        )

        run_task(task_cfg=task_cfg)

    def test_ci_lite(self):
        from evalscope.config import TaskConfig

        api_key = env.get('DASHSCOPE_API_KEY')

        task_cfg = TaskConfig(
            model='qwen-plus',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=api_key,
            eval_type=EvalType.SERVICE if api_key else EvalType.MOCK_LLM,
            datasets=[
                'general_mcq',
                'iquiz',
            ],
            dataset_args={
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',
                    'subset_list': [
                        'example'
                    ],
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',
                    'subset_list': [
                        'example'
                    ]
                }
            },
            eval_batch_size=1,
            limit=1,
            stream=True,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            judge_worker_num=1,
            judge_strategy=JudgeStrategy.AUTO,
            judge_model_args={
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
            }
        )

        run_task(task_cfg=task_cfg)
