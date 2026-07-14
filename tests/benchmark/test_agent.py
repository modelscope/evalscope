# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import sys
import tempfile
from dotenv import dotenv_values, load_dotenv
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

load_dotenv('.env')

env = dotenv_values('.env')

import unittest

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.claw_eval.claw_eval_adapter import ClawEvalAdapter
from evalscope.benchmarks.claw_eval.utils import (
    DEFAULT_CLAW_EVAL_SANDBOX_IMAGE,
    ClawEvalAssets,
    ensure_claw_eval_sandbox_image,
    load_claw_eval_trace,
    materialize_task_root,
    run_claw_eval_task,
    validate_claw_eval_private_api,
)
from evalscope.benchmarks.toolathlon.toolathlon_adapter import ToolathlonAdapter
from evalscope.config import SandboxTaskConfig, TaskConfig
from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


def _fake_claw_eval_scoring_modules() -> dict[str, ModuleType]:
    claw_eval = ModuleType('claw_eval')
    models = ModuleType('claw_eval.models')
    scoring = ModuleType('claw_eval.models.scoring')
    scoring.compute_pass_at_k = lambda scores, k: 1.0 if scores else 0.0
    scoring.compute_pass_hat_k = lambda scores, k: 1.0 if all(score >= 0.8 for score in scores) else 0.25
    return {
        'claw_eval': claw_eval,
        'claw_eval.models': models,
        'claw_eval.models.scoring': scoring,
    }


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
        config_overrides = {
            'collect_perf': False,
            'debug': False,
            'eval_batch_size': 1,
            'limit': 1,
            'no_timestamp': True,
            'work_dir': 'outputs/test_agent_browsecomp',
        }
        if not env.get('DASHSCOPE_API_KEY'):
            config_overrides['judge_strategy'] = JudgeStrategy.RULE

        self._run_dataset_test('browsecomp', **config_overrides)

        review_files = list(Path('outputs/test_agent_browsecomp').glob('reviews/*/browsecomp_default.jsonl'))
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

    def test_wide_search(self):
        """Test WideSearch with real qwen-plus, bash, Fetch MCP, and LLM judging."""
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is required for the WideSearch real-API smoke test.')

        self._run_dataset_test(
            'wide_search',
            model='qwen-plus',
            limit=1,
            eval_batch_size=1,
            collect_perf=False,
            debug=False,
            judge_model_args={
                'model_id': 'qwen-plus',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'extra_body': {'enable_thinking': False}
                }
            },
            agent_config=NativeAgentConfig(
                mcp_servers=[
                    MCPServerConfigStdio(
                        command=sys.executable,
                        args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                        name='fetch',
                    )
                ],
            ),
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

        with patch.object(ToolathlonAdapter, 'client_cls', FakeToolathlonClient):
            self._run_dataset_test(
                'toolathlon',
                dataset_args,
                use_mock=True,
                limit=1,
                eval_batch_size=1,
                no_timestamp=True,
                work_dir='outputs/test_agent_toolathlon',
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

    def test_claw_eval(self):
        """Test Claw-Eval task-level wrapper with mocked assets and private API runner."""
        run_calls = []

        def fake_run_claw_eval_task(**kwargs):
            run_calls.append(kwargs)
            self.assertTrue((kwargs['task_dir'] / 'task.yaml').is_file())
            self.assertEqual(kwargs['task_dir'].name, 'T002_task')
            self.assertTrue(kwargs['config_path'].is_file())
            return {
                'task_id': 'T002_task',
                'task_name': 'Task 2',
                'difficulty': 'easy',
                'trace_root': str(kwargs['trace_root']),
                'trace_path': str(kwargs['trace_root'] / f'T002_task_{len(run_calls)}.jsonl'),
                'raw_result': {
                    'task_id': 'T002_task'
                },
                'trials': [],
                'metrics': {
                    'avg_score': 0.8,
                    'task_score': 0.8,
                    'passed': 1.0,
                    'error_rate': 0.0,
                    'tokens': 10,
                    'model_input_tokens': 8,
                    'model_output_tokens': 2,
                    'wall_time_s': 1.5,
                    'model_time_s': 1.0,
                    'tool_time_s': 0.2,
                    'completion': 0.8,
                    'robustness': 0.8,
                    'communication': 0.0,
                    'safety': 1.0,
                },
            }

        def fake_manifest_loader(dataset_id, data_source, splits, force_redownload=False):
            self.assertEqual(dataset_id, 'claw-eval/Claw-Eval')
            self.assertEqual(splits, ['general'])
            return [
                {
                    'task_id': 'T001_task',
                    'split': 'general'
                },
                {
                    'task_id': 'T002_task',
                    'split': 'general'
                },
            ]

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / 'repo'
            tasks_dir = repo_root / 'tasks'
            (tasks_dir / 'T001_task').mkdir(parents=True)
            (tasks_dir / 'T002_task').mkdir(parents=True)
            (tasks_dir / 'T001_task' / 'task.yaml').write_text('id: T001_task\n', encoding='utf-8')
            (tasks_dir / 'T002_task' / 'task.yaml').write_text('id: T002_task\n', encoding='utf-8')
            fixtures_archive = Path(tmp) / 'fixtures.tar.gz'
            fixtures_archive.write_bytes(b'fixture')
            fixtures_dir = Path(tmp) / 'fixtures'
            fixtures_dir.mkdir()

            def fake_assets_preparer(**kwargs):
                self.assertNotIn('download_fixtures', kwargs)
                return ClawEvalAssets(
                    repo_root=repo_root,
                    tasks_dir=tasks_dir,
                    fixtures_archive=fixtures_archive,
                    fixtures_dir=fixtures_dir,
                )

            prepared_images = []

            def fake_image_preparer(repo_root_arg):
                prepared_images.append(repo_root_arg)
                return DEFAULT_CLAW_EVAL_SANDBOX_IMAGE

            dataset_args = {
                'subset_list': ['general'],
                'extra_params': {
                    'task_ids': ['T002_task'],
                }
            }

            with patch('evalscope.benchmarks.claw_eval.claw_eval_adapter.run_claw_eval_task',
                       side_effect=fake_run_claw_eval_task), \
                    patch('evalscope.benchmarks.claw_eval.claw_eval_adapter.load_task_manifest',
                          side_effect=fake_manifest_loader), \
                    patch('evalscope.benchmarks.claw_eval.claw_eval_adapter.prepare_claw_eval_assets',
                          side_effect=fake_assets_preparer), \
                    patch('evalscope.benchmarks.claw_eval.claw_eval_adapter.ensure_claw_eval_sandbox_image',
                          side_effect=fake_image_preparer), \
                    patch.dict(sys.modules, _fake_claw_eval_scoring_modules()), \
                    patch.object(ClawEvalAdapter, '_check_runtime', return_value=None):
                self._run_dataset_test(
                    'claw_eval',
                    dataset_args,
                    use_mock=True,
                    limit=1,
                    eval_batch_size=1,
                    no_timestamp=True,
                    work_dir='outputs/test_agent_claw_eval',
                    api_url='http://localhost:8000/v1',
                    api_key='local-key',
                    repeats=3,
                )

        self.assertEqual(len(run_calls), 3)
        self.assertEqual(run_calls[0]['model_id'], 'qwen3-max')
        self.assertEqual(run_calls[0]['api_key'], 'local-key')
        self.assertEqual(run_calls[0]['base_url'], 'http://localhost:8000/v1')
        self.assertEqual([call['port_offset'] for call in run_calls], [0, 0, 0])
        self.assertTrue(all(str(call['repo_root']).endswith('/repo') for call in run_calls))
        self.assertEqual(prepared_images, [repo_root])
        review_files = list(Path('outputs/test_agent_claw_eval').glob('reviews/*/claw_eval_general.jsonl'))
        self.assertEqual(len(review_files), 1)
        reviews = [json.loads(line) for line in review_files[0].read_text(encoding='utf-8').splitlines() if line]
        self.assertEqual(len(reviews), 3)
        review = reviews[0]
        metadata = review['sample_score']['sample_metadata']
        self.assertEqual(metadata['split'], 'general')
        self.assertEqual(metadata['task_id'], 'T002_task')
        self.assertEqual({row['sample_score']['group_id'] for row in reviews}, {0})
        self.assertNotIn('selected_task_ids', metadata)
        self.assertNotIn('selected_records', metadata)

    def test_claw_eval_init_checks_runtime(self):
        """Test Claw-Eval fails early when the official package is unavailable."""
        with patch.object(ClawEvalAdapter, '_check_runtime') as check:
            adapter = get_benchmark('claw_eval', TaskConfig(datasets=['claw_eval']))

        self.assertIsInstance(adapter, ClawEvalAdapter)
        check.assert_called_once_with()

    def test_claw_eval_runner_uses_official_sandbox(self):
        """Test the runner always invokes the official private API sandbox path."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_root = root / 'traces'
            config_path = root / 'config.yaml'
            config_path.write_text('sandbox:\n  enabled: true\n', encoding='utf-8')
            task_dir = root / 'tasks' / 'T001_task'
            task_dir.mkdir(parents=True)
            (task_dir / 'task.yaml').write_text('id: T001_task\n', encoding='utf-8')
            with patch('evalscope.benchmarks.claw_eval.utils.validate_claw_eval_private_api', return_value=None), \
                    patch('claw_eval.cli._run_single_task', return_value={
                        'task_id': 'T001_task',
                        'task_name': 'Task 1',
                        'difficulty': 'easy',
                        'trials': [{
                            'trace': str(trace_root / 'T001_task_abc.jsonl'),
                            'task_score': 1.0,
                            'passed': True,
                        }],
                        'error': None,
                    }) as run:
                parsed = run_claw_eval_task(
                    task_dir=task_dir,
                    trace_root=trace_root,
                    config_path=config_path,
                    repo_root=root,
                    model_id='qwen3-max',
                    api_key='local-key',
                    base_url='http://localhost:8000/v1',
                    port_offset=50,
                )

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs['task_dir'], str(task_dir.resolve()))
        self.assertEqual(kwargs['trials'], 1)
        self.assertEqual(kwargs['port_offset'], 50)
        self.assertTrue(kwargs['sandbox'])
        self.assertFalse(kwargs['sandbox_tools'])
        self.assertIsNone(kwargs['sandbox_image'])
        self.assertEqual(kwargs['api_key'], 'local-key')
        self.assertEqual(parsed['metrics']['avg_score'], 1.0)

    def test_claw_eval_private_api_signature_guard(self):
        """Test incompatible Claw-Eval private API signatures fail fast."""

        def incompatible_run_single_task(task_dir):
            return {}

        with patch('claw_eval.cli._run_single_task', incompatible_run_single_task):
            with self.assertRaisesRegex(RuntimeError, 'private API is incompatible'):
                validate_claw_eval_private_api()

    def test_claw_eval_aggregate_scores_groups_repeats(self):
        """Test grouped Claw-Eval repeat aggregation."""
        adapter = object.__new__(ClawEvalAdapter)
        scores = [
            SampleScore(score=Score(value={'avg_score': 1.0}, metadata={}), sample_id=0, group_id=0),
            SampleScore(score=Score(value={'avg_score': 0.0}, metadata={}), sample_id=1, group_id=0),
            SampleScore(score=Score(value={'avg_score': 0.5}, metadata={'error': 'failed'}), sample_id=2, group_id=1),
            SampleScore(score=Score(value={'avg_score': 1.0}, metadata={}), sample_id=3, group_id=1),
        ]

        with patch.dict(sys.modules, _fake_claw_eval_scoring_modules()):
            aggregated = {(score.metric_name, score.aggregation_name): score.score
                          for score in adapter.aggregate_scores(scores)}

        self.assertAlmostEqual(aggregated[('avg_score', 'mean')], 0.625)
        self.assertAlmostEqual(aggregated[('pass_at_k', 'mean')], 1.0)
        self.assertAlmostEqual(aggregated[('pass_hat_k', 'mean')], 0.25)
        self.assertAlmostEqual(aggregated[('error_rate', 'mean')], 0.25)

    def test_claw_eval_trace_converts_to_agent_trace(self):
        """Test official Claw-Eval JSONL traces are exposed as EvalScope agent traces."""
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / 'trace.jsonl'
            rows = [
                {
                    'type': 'trace_start',
                    'trace_id': 'trace-1',
                    'timestamp': '2026-07-14T00:00:00+00:00',
                },
                {
                    'type': 'message',
                    'message': {
                        'role': 'user',
                        'content': [{
                            'type': 'text',
                            'text': 'Sort my inbox.'
                        }],
                    },
                    'usage': {
                        'input_tokens': 0,
                        'output_tokens': 0
                    },
                    'timestamp': '2026-07-14T00:00:00+00:00',
                },
                {
                    'type': 'message',
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Checking email.'
                            },
                            {
                                'type': 'tool_use',
                                'id': 'call-1',
                                'name': 'gmail_list_messages',
                                'input': {
                                    'max_results': 5
                                },
                            },
                        ],
                    },
                    'usage': {
                        'input_tokens': 10,
                        'output_tokens': 3
                    },
                    'timestamp': '2026-07-14T00:00:01+00:00',
                },
                {
                    'type': 'tool_dispatch',
                    'tool_use_id': 'call-1',
                    'tool_name': 'gmail_list_messages',
                    'request_body': {
                        'max_results': 5
                    },
                    'response_status': 200,
                    'latency_ms': 2.5,
                    'timestamp': '2026-07-14T00:00:01+00:00',
                },
                {
                    'type': 'message',
                    'message': {
                        'role': 'user',
                        'content': [{
                            'type': 'tool_result',
                            'tool_use_id': 'call-1',
                            'content': [{
                                'type': 'text',
                                'text': '{"messages": []}'
                            }],
                            'is_error': False,
                        }],
                    },
                    'timestamp': '2026-07-14T00:00:02+00:00',
                },
                {
                    'type': 'trace_end',
                    'trace_id': 'trace-1',
                    'input_tokens': 10,
                    'output_tokens': 3,
                    'total_tokens': 13,
                    'wall_time_s': 2.0,
                    'timestamp': '2026-07-14T00:00:03+00:00',
                },
            ]
            trace_path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')

            trace, messages = load_claw_eval_trace(str(trace_path))

        self.assertIsNotNone(trace)
        self.assertIsNotNone(messages)
        self.assertEqual(trace.framework, 'claw-eval')
        self.assertEqual(trace.trial_id, 'trace-1')
        self.assertEqual(trace.total_usage.total_tokens, 13)
        self.assertEqual([message.role for message in messages], ['user', 'assistant', 'tool'])
        self.assertEqual(messages[1].tool_calls[0].function.name, 'gmail_list_messages')
        self.assertIn('tool_result', {event.type.value for event in trace.events})

    def test_claw_eval_image_auto_build(self):
        """Test missing official sandbox image is built from the official repo."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / 'Dockerfile.agent').write_text('FROM scratch\n', encoding='utf-8')
            with patch('evalscope.benchmarks.claw_eval.utils.should_build_docker_image', return_value=True), \
                    patch('evalscope.benchmarks.claw_eval.utils.build_docker_image') as build:
                image = ensure_claw_eval_sandbox_image(repo_root)

        self.assertEqual(image, DEFAULT_CLAW_EVAL_SANDBOX_IMAGE)
        build.assert_called_once_with(
            image=DEFAULT_CLAW_EVAL_SANDBOX_IMAGE,
            path=str(repo_root),
            dockerfile='Dockerfile.agent',
        )

    def test_claw_eval_task_root_keeps_fixture_cross_refs(self):
        """Test selected task roots keep non-selected fixture-only cross refs."""
        with tempfile.TemporaryDirectory() as tmp:
            source_tasks = Path(tmp) / 'source' / 'tasks'
            source_mock_services = source_tasks.parent / 'mock_services'
            selected_task = source_tasks / 'T001_task'
            fixture_only_task = source_tasks / 'T999_fixture_source'
            selected_task.mkdir(parents=True)
            source_mock_services.mkdir(parents=True)
            (selected_task / 'task.yaml').write_text('id: T001_task\n', encoding='utf-8')
            (source_mock_services / 'server.py').write_text('print("ok")\n', encoding='utf-8')
            (fixture_only_task / 'fixtures').mkdir(parents=True)
            (fixture_only_task / 'fixtures' / 'data.json').write_text('{}\n', encoding='utf-8')

            runtime_tasks = materialize_task_root(
                source_tasks_dir=source_tasks,
                selected_task_ids=['T001_task'],
                output_root=Path(tmp) / 'runtime',
            )

            self.assertTrue((runtime_tasks / 'T001_task' / 'task.yaml').is_file())
            self.assertTrue((Path(tmp) / 'runtime' / 'mock_services' / 'server.py').is_file())
            self.assertTrue((runtime_tasks / 'T999_fixture_source' / 'fixtures' / 'data.json').is_file())
            self.assertFalse((runtime_tasks / 'T999_fixture_source' / 'task.yaml').exists())

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
