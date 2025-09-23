# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import unittest

from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestCodeBenchmark(TestBenchmark):
    """Benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen-plus',
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
            'use_sandbox': True,
            'sandbox_type': 'docker',
            'debug': True,
        }

    def test_humaneval(self):
        """Test Humaneval dataset."""
        self._run_dataset_test('humaneval', limit=5)

    def test_humaneval_remote_sandbox(self):
        """Test Humaneval dataset with remote sandbox manager."""
        sandbox_manager_config = {'base_url': 'http://localhost:8000'}
        self._run_dataset_test('humaneval', limit=5, sandbox_manager_config=sandbox_manager_config)

    def test_live_code_bench(self):
        """Test Live Code Bench dataset."""
        dataset_args = {
            'subset_list': ['v5'],
            'review_timeout': 6,
            'extra_params': {
                'start_date': '2024-08-01',
                'end_date': '2025-02-28'
            },
        }
        self._run_dataset_test('live_code_bench', limit=5, dataset_args=dataset_args, use_cache='outputs/20250918_200232', rerun_review=True)

    def test_live_code_bench_remote_sandbox(self):
        """Test Live Code Bench dataset."""
        dataset_args = {
            'subset_list': ['v5'],
            'review_timeout': 6,
            'extra_params': {
                'start_date': '2024-08-01',
                'end_date': '2025-02-28'
            },
        }
        sandbox_manager_config = {'base_url': 'http://localhost:8000'}
        self._run_dataset_test('live_code_bench', limit=20, dataset_args=dataset_args, sandbox_manager_config=sandbox_manager_config, use_cache='outputs/20250918_200232_2', rerun_review=True)
