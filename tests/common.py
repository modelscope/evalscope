# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import unittest
from unittest import TestCase

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()


class TestBenchmark(TestCase):
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
            'debug': True,
        }

    def _run_dataset_test(self, dataset_name, dataset_args=None, use_mock=False, **config_overrides):
        """Helper method to run test for a specific dataset."""
        config = self.base_config.copy()
        config['datasets'] = [dataset_name]

        if not env.get('DASHSCOPE_API_KEY'):
            use_mock = True
            logger.warning('DASHSCOPE_API_KEY is not set. Using mock evaluation.')

        if use_mock:
            config['eval_type'] = EvalType.MOCK_LLM

        # 应用配置覆盖
        config.update(config_overrides)

        if dataset_args:
            config['dataset_args'] = {dataset_name: dataset_args}

        task_cfg = TaskConfig(**config)
        run_task(task_cfg=task_cfg)

    def _run_dataset_load_test(self, dataset_name, dataset_args=None, **config_overrides):
        """Helper method to test dataset loading."""

        self._run_dataset_test(dataset_name, dataset_args, use_mock=True, limit=None, **config_overrides)
