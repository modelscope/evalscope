# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values, load_dotenv

load_dotenv('.env')

env = dotenv_values('.env')

import unittest

from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestModels(TestBenchmark):
    """Benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'Qwen2.5-VL-7B-Instruct',
            'api_url': 'http://localhost:8000/',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.ANTHROPIC_API,  # test anthropic api
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 4096,
                'temperature': 0.7,
                'parallel_tool_calls': True,
                'stream': True
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_model_args': {
                'model_id': 'qwen3-235b-a22b',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096,
                    'extra_body': {'enable_thinking': False}
                }
            },
            'debug': True,
        }


    # Math & Reasoning datasets
    def test_gsm8k(self):
        """Test GSM8K math reasoning dataset."""
        dataset_args = {
            'few_shot_num': 0,
        }
        self._run_dataset_test('gsm8k', dataset_args=dataset_args, limit=5)

    def test_chartqa(self):
        """Test ChartQA math reasoning dataset."""
        dataset_args = {
            'few_shot_num': 0,
        }
        self._run_dataset_test('chartqa', dataset_args=dataset_args, limit=5)

    def test_bfcl_v4(self):
        """Test BFCL-V4 math reasoning dataset."""
        """Test BFCL v4 dataset."""
        dataset_args = {
            'subset_list': [
                'simple_python',
                # 'simple_java',
                # 'simple_javascript',
                # 'multiple',
                # 'parallel',
                # 'parallel_multiple',
                # 'irrelevance',
                # 'live_simple',
                # 'live_multiple',
                # 'live_parallel',
                # 'live_parallel_multiple',
                # 'live_irrelevance',
                # 'live_relevance',
                # 'multi_turn_base',
                # 'multi_turn_miss_func',
                # 'multi_turn_miss_param',
                # 'multi_turn_long_context',
                # 'web_search_base',
                # 'web_search_no_snippet',
                # 'memory_kv',
                # 'memory_vector',
                # 'memory_rec_sum'
            ],
            'extra_params': {
                'is_fc_model': True,
                'underscore_to_dot': True,
                'SERPAPI_API_KEY':env.get('SERPAPI_API_KEY'),
            }
        }
        self._run_dataset_test('bfcl_v4', dataset_args=dataset_args)

    def test_chartqa(self):
        """Test ChartQA math reasoning dataset."""

        self._run_dataset_test('chartqa')
