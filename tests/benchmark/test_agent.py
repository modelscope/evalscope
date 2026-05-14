# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values, load_dotenv

load_dotenv('.env')

env = dotenv_values('.env')

import unittest

from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestAgentBenchmark(TestBenchmark):
    """Agentic benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.OPENAI_API,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'temperature': 0.7,
                'parallel_tool_calls': True,
                'retries': 3,
                'extra_body': {'enable_thinking': True},
                'stream': True
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_model_args': {
                'model_id': 'qwen3-max',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'extra_body': {'enable_thinking': False}
                }
            },
            'debug': True,
        }

    def test_swe_bench_verified_agentic(self):
        """Test SWE-bench-verified agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
                'action_protocol': 'toolcall',
                'max_steps': 250,
                'command_timeout': 60.0,
                'working_dir': '/testbed',
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test('swe_bench_verified_agentic', dataset_args, limit=1)

    def test_swe_bench_verified_mini_agentic(self):
        """Test SWE-bench-verified-mini agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
                'action_protocol': 'toolcall',
                'max_steps': 250,
                'command_timeout': 60.0,
                'working_dir': '/testbed',
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test('swe_bench_verified_mini_agentic', dataset_args, limit=1)

    def test_swe_bench_lite_agentic(self):
        """Test SWE-bench-lite agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
                'action_protocol': 'toolcall',
                'max_steps': 250,
                'command_timeout': 60.0,
                'working_dir': '/testbed',
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test('swe_bench_lite_agentic', dataset_args, limit=1)

    def test_swe_bench_verified_agentic_backticks(self):
        """Test SWE-bench-verified agentic dataset with backticks protocol."""
        dataset_args = {
            'extra_params': {
                'action_protocol': 'backticks',
                'max_steps': 250,
                'command_timeout': 60.0,
                'working_dir': '/testbed',
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test('swe_bench_verified_agentic', dataset_args, limit=1)


if __name__ == '__main__':
    # Run specific test: python -m unittest test_agent.TestAgentBenchmark.test_swe_bench_verified_agentic
    # Run all tests: python -m unittest test_agent.TestAgentBenchmark
    unittest.main()
