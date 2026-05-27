# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values, load_dotenv

load_dotenv('.env')

env = dotenv_values('.env')

import unittest

from evalscope.config import SandboxTaskConfig
from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestAgentBenchmark(TestBenchmark):
    """Agentic benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen3-max',
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
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test('swe_bench_verified_mini_agentic', dataset_args, limit=3)

    def test_swe_bench_lite_agentic(self):
        """Test SWE-bench-lite agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
                'action_protocol': 'toolcall',
                'max_steps': 250,
                'command_timeout': 60.0,
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test('swe_bench_lite_agentic', dataset_args, limit=1)

    def test_swe_bench_pro(self):
        """Test SWE-bench_Pro agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
                'action_protocol': 'toolcall',
                'max_steps': 250,
                'command_timeout': 60.0,
                'eval_timeout': 1800,
            }
        }
        self._run_dataset_test(
            'swe_bench_pro',
            dataset_args,
            limit=5,
            use_cache='outputs/20260519_155200',
            rerun_review=True,
            sandbox=SandboxTaskConfig(
                default_config={'platform': 'linux/amd64', 'memory_limit': '12g', 'cpu_limit': 4.0},
            ),
        )

    def test_gaia(self):
        """Test GAIA benchmark using docker environment with react + bash."""
        dataset_args = {
            'subset_list': ['2023_level1', '2023_level2', '2023_level3'],
            'extra_params': {
                'max_steps': 50,
                'command_timeout': 180.0,
                'docker_image': 'python:3.11',
                'network_enabled': True,
            }
        }
        self._run_dataset_test('gaia', dataset_args, limit=1)

    def test_gaia_with_mcp(self):
        """GAIA + MCP fetch server, exercising the host-side MCP plumbing.

        Requires ``pip install mcp-server-fetch`` in the eval environment.
        Using ``python -m mcp_server_fetch`` (rather than ``uvx``) keeps the
        test deterministic — no per-run package fetch / venv creation.
        """
        import sys

        from evalscope.api.agent import NativeAgentConfig
        from evalscope.api.agent.mcp import MCPServerConfigStdio
        dataset_args = {
            'subset_list': ['2023_level1'],
            'extra_params': {
                'max_steps': 30,
                'command_timeout': 180.0,
                'docker_image': 'python:3.11',
                'network_enabled': True,
            }
        }
        agent_config = NativeAgentConfig(mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                # ``--ignore-robots-txt`` lets the server fetch sites whose
                # robots.txt is unreachable (transient network failures /
                # CDN-blocked UAs commonly seen during offline-ish CI runs).
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
        ])
        self._run_dataset_test('gaia', dataset_args, limit=1, agent_config=agent_config)

    def test_terminal_bench_v2_1(self):
        """Test Terminal-Bench v2.1 dataset."""
        dataset_args = {
            'extra_params': {
                'timeout_multiplier': 3,
                'environment_kwargs': {'override_cpus': 2},
            },
        }
        self._run_dataset_test('terminal_bench_v2_1', dataset_args, limit=3, eval_batch_size=3)

    def test_swe_bench_verified_agentic_backticks(self):
        """Test SWE-bench-verified agentic dataset with backticks protocol."""
        dataset_args = {
            'extra_params': {
                'action_protocol': 'backticks',
                'max_steps': 250,
                'command_timeout': 60.0,
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
