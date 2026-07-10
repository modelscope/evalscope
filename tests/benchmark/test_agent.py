# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import sys
import tempfile
from dotenv import dotenv_values, load_dotenv
from pathlib import Path
from unittest.mock import patch

load_dotenv('.env')

env = dotenv_values('.env')

import unittest

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio
from evalscope.benchmarks.toolathlon.toolathlon_adapter import ToolathlonAdapter
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

    def test_browsecomp(self):
        """Test BrowseComp benchmark end-to-end."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_overrides = {
                'collect_perf': False,
                'debug': False,
                'eval_batch_size': 1,
                'limit': 1,
                'no_timestamp': True,
                'work_dir': tmp_dir,
            }
            if not env.get('DASHSCOPE_API_KEY'):
                config_overrides['judge_strategy'] = JudgeStrategy.RULE

            self._run_dataset_test('browsecomp', **config_overrides)

            review_files = list(Path(tmp_dir).glob('reviews/*/browsecomp_default.jsonl'))
            self.assertEqual(len(review_files), 1)
            review = json.loads(review_files[0].read_text(encoding='utf-8').strip())
            self.assertNotIn('canary', review['sample_score']['sample_metadata'])

    def test_swe_bench_verified_agentic(self):
        """Test SWE-bench-verified agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
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
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test('swe_bench_lite_agentic', dataset_args, limit=1)

    def test_swe_bench_multilingual_agentic(self):
        """Test SWE-bench-multilingual agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
                'build_docker_images': False,
                'pull_remote_images_if_available': True,
            }
        }
        self._run_dataset_test(
            'swe_bench_multilingual_agentic',
            dataset_args,
            limit=1,
            generation_config={
                'temperature': 0.0,
                'parallel_tool_calls': False,
                'retries': 3,
                'extra_body': {'enable_thinking': True},
                'stream': True
            },
        )

    def test_swe_bench_pro(self):
        """Test SWE-bench_Pro agentic dataset using docker environment."""
        dataset_args = {
            'extra_params': {
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
        }
        self._run_dataset_test(
            'gaia',
            dataset_args,
            limit=1,
            sandbox=SandboxTaskConfig(default_config={'image': 'python:3.11', 'network_enabled': True}),
        )

    def test_gaia_with_mcp(self):
        """GAIA + MCP fetch server, exercising the host-side MCP plumbing.

        Requires ``pip install mcp-server-fetch`` in the eval environment.
        Using ``python -m mcp_server_fetch`` (rather than ``uvx``) keeps the
        test deterministic — no per-run package fetch / venv creation.
        """
        dataset_args = {
            'subset_list': ['2023_level1'],
        }
        agent_config = NativeAgentConfig(
            max_steps=30,
            mcp_servers=[
                MCPServerConfigStdio(
                    command=sys.executable,
                    # ``--ignore-robots-txt`` lets the server fetch sites whose
                    # robots.txt is unreachable (transient network failures /
                    # CDN-blocked UAs commonly seen during offline-ish CI runs).
                    args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                    name='fetch',
                ),
            ],
        )
        self._run_dataset_test(
            'gaia',
            dataset_args,
            limit=1,
            agent_config=agent_config,
            sandbox=SandboxTaskConfig(default_config={'image': 'python:3.11', 'network_enabled': True}),
        )

    def test_researchrubrics(self):
        """Test ResearchRubrics with a real agent API and binary LLM judge."""
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is required for the ResearchRubrics real-API smoke test.')

        self._run_dataset_test(
            'researchrubrics',
            limit=5,
            eval_batch_size=5,
            collect_perf=False,
            debug=False,
        )

    def test_terminal_bench_v2_1(self):
        """Test Terminal-Bench v2.1 dataset."""
        dataset_args = {
            'extra_params': {
                'timeout_multiplier': 3,
                'environment_kwargs': {'override_cpus': 2},
            },
        }
        self._run_dataset_test('terminal_bench_v2_1', dataset_args, limit=3, eval_batch_size=3)

    def test_toolathlon(self):
        """Test Toolathlon official-service wrapper with a mocked service client."""

        class FakeToolathlonClient:
            last_config = None

            def __init__(self, config):
                FakeToolathlonClient.last_config = config

            def run_private(self):
                return {
                    'job_id': self.last_config.job_id or 'fake-toolathlon-job',
                    'output_dir': str(self.last_config.output_dir),
                    'acc': 1.0,
                    'eval_stats': {
                        'passed': 1,
                        'total': 1
                    },
                    'task_results': [{
                        'task': 'find-alita-paper',
                        'pass': True
                    }],
                }

        dataset_args = {
            'extra_params': {
                'task_list': ['find-alita-paper'],
                'workers': 1,
                'skip_container_restart': True,
                'model_params': {
                    'temperature': 0.0
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(ToolathlonAdapter, 'client_cls', FakeToolathlonClient):
                self._run_dataset_test(
                    'toolathlon',
                    dataset_args,
                    use_mock=True,
                    limit=1,
                    eval_batch_size=1,
                    no_timestamp=True,
                    work_dir=tmp_dir,
                    api_url='http://localhost:8000/v1',
                    api_key='local-key',
                )

            self.assertIsNotNone(FakeToolathlonClient.last_config)
            self.assertEqual(FakeToolathlonClient.last_config.base_url, 'http://localhost:8000/v1')
            self.assertEqual(FakeToolathlonClient.last_config.api_key, 'local-key')
            self.assertEqual(FakeToolathlonClient.last_config.task_list, ['find-alita-paper'])
            self.assertEqual(FakeToolathlonClient.last_config.model_params['temperature'], 0.0)
            self.assertNotIn('retries', FakeToolathlonClient.last_config.model_params)
            self.assertNotIn('batch_size', FakeToolathlonClient.last_config.model_params)

    def test_swe_bench_verified_agentic_backticks(self):
        """Test SWE-bench-verified agentic dataset with backticks protocol."""
        dataset_args = {
            'extra_params': {
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
                'force_arch': 'arm64',
            }
        }
        self._run_dataset_test(
            'swe_bench_verified_agentic',
            dataset_args,
            limit=1,
            agent_config=NativeAgentConfig(strategy='swe_bench_backticks'),
        )

if __name__ == '__main__':
    # Run specific test: python -m unittest test_agent.TestAgentBenchmark.test_swe_bench_verified_agentic
    # Run all tests: python -m unittest test_agent.TestAgentBenchmark
    unittest.main()
