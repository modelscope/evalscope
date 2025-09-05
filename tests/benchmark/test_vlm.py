# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import unittest

from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestVLMBenchmark(TestBenchmark):
    """Benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen-vl-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.SERVICE,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 4096,
                'temperature': 0.0,
                'seed': 42,
                'parallel_tool_calls': True
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_worker_num': 5,
            'judge_model_args': {
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096,
                }
            },
            'debug': True,
        }

    def test_mmmu(self):
        dataset_args = {
            'subset_list':[
                'Accounting',
                'Agriculture',
                # 'Architecture_and_Engineering'
            ]
        }
        self._run_dataset_test('mmmu', dataset_args=dataset_args)

    def test_math_vista(self):
        dataset_args = {
            'subset_list': ['default']
        }
        self._run_dataset_test('math_vista', dataset_args=dataset_args)

    def test_mmmu_pro(self):
        dataset_args = {
            'subset_list':[
                'Accounting',
                # 'Agriculture',
            ],
            'extra_params': {
                'dataset_format': 'standard (4 options)',  # 'standard (4 options)', 'standard (10 options)', 'vision'
            },
        }
        self._run_dataset_test('mmmu_pro', dataset_args=dataset_args, limit=10)

    def test_qwen3_collection(self):
        dataset_args = {
            'dataset_id': 'outputs/qwen3_vl_test.jsonl',
            'shuffle': True,
        }
        self._run_dataset_test('data_collection', dataset_args)
