from __future__ import annotations

import asyncio
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock

import pytest

from evalscope.agent.external.config import ExternalAgentConfig
from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult, NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.skillsbench.skillsbench_adapter import SkillsBenchAdapter
from evalscope.benchmarks.skillsbench.utils import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_WITH_SKILL,
    skillsbench_sandbox_config,
    stage_environment_context,
)
from evalscope.config import TaskConfig
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


def test_stage_no_skill_removes_skills_and_copy_lines(tmp_path: Path) -> None:
    task = _make_task_env(tmp_path)

    context = Path(stage_environment_context(task_dir=task, skill_mode=SKILL_MODE_NO_SKILL))

    try:
        assert not (context / 'skills').exists()
        dockerfile = (context / 'Dockerfile').read_text(encoding='utf-8')
        assert 'COPY skills' not in dockerfile
        instruction_lines = [line for line in dockerfile.splitlines() if not line.lstrip().startswith('#')]
        assert all('/skills' not in line for line in instruction_lines)
    finally:
        _cleanup(context)


def test_stage_with_skill_creates_empty_skills_dir_when_missing(tmp_path: Path) -> None:
    task = _make_task_env(tmp_path)
    _cleanup(task / 'environment' / 'skills')

    context = Path(stage_environment_context(task_dir=task, skill_mode=SKILL_MODE_WITH_SKILL))

    try:
        assert (context / 'skills').is_dir()
        assert 'COPY skills /skills' in (context / 'Dockerfile').read_text(encoding='utf-8')
    finally:
        _cleanup(context)


def test_skillsbench_sandbox_config_uses_task_image_workdir_and_network() -> None:
    config = skillsbench_sandbox_config(
        {
            'image_tag': 'evalscope-skillsbench-demo:abc',
            'working_dir': '/app',
            'frontmatter': {'environment': {'network_mode': 'no-network'}},
        }
    )

    assert config == {
        'image': 'evalscope-skillsbench-demo:abc',
        'command': 'sleep infinity',
        'working_dir': '/app',
        'tools_config': ['shell_executor'],
        'network_enabled': False,
    }


def test_select_task_dirs_ignores_hidden_directories(tmp_path: Path) -> None:
    visible = _make_task_env(tmp_path)
    hidden = tmp_path / '.ipynb_checkpoints'
    hidden.mkdir()

    adapter = SkillsBenchAdapter(
        benchmark_meta=BenchmarkMeta(
            name='skillsbench',
            dataset_id='skillsbench',
            subset_list=['default'],
            metric_list=['score'],
            prompt_template='{question}',
        ),
        task_config=TaskConfig(datasets=['skillsbench']),
    )
    adapter.tasks_dir = tmp_path

    assert adapter._select_task_dirs() == [visible]


def test_run_verifier_records_timeout_before_reward_parsing(tmp_path: Path, monkeypatch) -> None:
    task = tmp_path / 'task'
    (task / 'verifier').mkdir(parents=True)
    adapter = SkillsBenchAdapter(
        benchmark_meta=BenchmarkMeta(
            name='skillsbench',
            dataset_id='skillsbench',
            subset_list=['default'],
            metric_list=['score'],
            prompt_template='{question}',
        ),
        task_config=TaskConfig(datasets=['skillsbench']),
    )
    adapter.runner = 'oracle'
    sample = Sample(
        id=0,
        input='do task',
        target='',
        metadata={
            'task_id': 'task/one',
            'task_dir': str(task),
            'skill_mode': 'with-skill',
            'agent_timeout_sec': 1,
            'verifier_timeout_sec': 1,
        },
    )
    monkeypatch.setattr(adapter, '_build_environment', lambda _: _TimeoutVerifierEnv())
    adapter.run_inference(None, sample, str(tmp_path))

    assert sample.metadata['verifier_timed_out'] is True
    assert sample.metadata['verifier_error'] == 'Verifier timed out'
    assert sample.metadata['reward'] == 0.0
    assert Path(sample.metadata['verifier_stdout_path']).read_text(encoding='utf-8') == 'partial stdout'
    assert 'ctrf_path' not in sample.metadata


@pytest.mark.parametrize('shared_root', [True, False], ids=['same-output-dir', 'separate-output-dirs'])
@pytest.mark.parametrize('finish_first', [0, 1], ids=['first-finishes-first', 'second-finishes-first'])
def test_concurrent_oracle_runs_keep_verifier_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_root: bool, finish_first: int
) -> None:
    config = TaskConfig(datasets=['skillsbench'], dataset_args={'skillsbench': {'extra_params': {'runner': 'oracle'}}})
    adapter = get_benchmark('skillsbench', config)
    started = [Event(), Event()]
    release = [Event(), Event()]
    roots = [tmp_path / 'first', tmp_path / ('first' if shared_root else 'second')]
    rewards = [0.5, 1.0]

    def build_environment(sample: Sample) -> AgentEnvironment:
        async def exec_command(cmd: list[str], **kwargs) -> ExecResult:
            if cmd[-1] == '/oracle/solve.sh':
                started[sample.id].set()
                assert await asyncio.to_thread(release[sample.id].wait, 10)
            if cmd[-1].startswith('cat '):
                filename = Path(shlex.split(cmd[-1])[1]).name
                files = {
                    'test-stdout.txt': f'log for task {sample.id}',
                    'reward.txt': str(rewards[sample.id]),
                    'ctrf.json': json.dumps({'task': sample.id}),
                }
                return ExecResult(stdout=files[filename])
            return ExecResult()

        env = AsyncMock(spec=AgentEnvironment)
        env.exec.side_effect = exec_command
        return env

    def run(task_id: int) -> TaskState:
        sample = Sample(
            id=task_id,
            input='run oracle',
            metadata={
                'task_id': f'task/{task_id}', 'task_dir': str(tmp_path), 'skill_mode': 'with-skill',
                'agent_timeout_sec': 10, 'verifier_timeout_sec': 10,
            },
        )
        try:
            return adapter.run_inference(None, sample, str(roots[task_id]))
        finally:
            AsyncioLoopRunner.shutdown_for_thread()

    monkeypatch.setattr(adapter, '_build_environment', build_environment)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            futures = []
            for i in range(2):
                futures.append(pool.submit(run, i))
                assert started[i].wait(10)
            release[finish_first].set()
            futures[finish_first].result(timeout=10)
        finally:
            for event in release:
                event.set()

    for i, future in enumerate(futures):
        state = future.result()
        score = adapter.match_score('', '', '', state)
        assert score.value['normalized_score'] == rewards[i]
        directory = roots[i] / 'artifacts' / 'skillsbench' / f'task-{i}' / 'with-skill'
        assert score.metadata['artifact_dir'] == str(directory)
        for key, name, contents in [
            ('verifier_stdout_path', 'test-stdout.txt', f'log for task {i}'),
            ('reward_path', 'reward.txt', str(rewards[i])),
            ('ctrf_path', 'ctrf.json', json.dumps({'task': i})),
        ]:
            assert score.metadata[key] == str(directory / name)
            assert (directory / name).read_text(encoding='utf-8') == contents


def test_on_inference_accepts_native_agent_config(monkeypatch) -> None:
    adapter = SkillsBenchAdapter(
        benchmark_meta=BenchmarkMeta(
            name='skillsbench',
            dataset_id='skillsbench',
            subset_list=['default'],
            metric_list=['score'],
            prompt_template='{question}',
        ),
        task_config=TaskConfig(
            datasets=['skillsbench'],
            agent_config=NativeAgentConfig(tools=['bash']),
        ),
    )
    env = _FakeSkillsBenchEnv()
    sample = Sample(id=0, input='do task', target='', metadata={'task_id': 'task/one'})

    def fake_run_native_agent(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs['environment_override'] is env
        output = ModelOutput(
            model='fake',
            choices=[ChatCompletionChoice.from_content('done')],
            metadata={'source': 'test'},
        )
        return InferenceResult(
            output=output,
            messages=[ChatMessageAssistant(content='done')],
        )

    async def fake_run_verifier(run_env, run_sample):  # type: ignore[no-untyped-def]
        assert run_env is env
        run_sample.metadata['reward'] = 1.0

    monkeypatch.setattr(adapter, '_build_environment', lambda _: env)
    monkeypatch.setattr('evalscope.agent.runner.run_native_agent', fake_run_native_agent)
    monkeypatch.setattr(adapter, '_run_verifier', fake_run_verifier)

    result = adapter._on_inference(model=None, sample=sample)

    assert result.output.message.text == 'done'
    assert sample.metadata['reward'] == 1.0
    assert env.closed


def test_on_inference_keeps_external_environment_open_until_verifier(monkeypatch) -> None:
    adapter = SkillsBenchAdapter(
        benchmark_meta=BenchmarkMeta(
            name='skillsbench',
            dataset_id='skillsbench',
            subset_list=['default'],
            metric_list=['score'],
            prompt_template='{question}',
        ),
        task_config=TaskConfig(
            datasets=['skillsbench'],
            agent_config=ExternalAgentConfig(framework='codex', timeout=60),
        ),
    )
    env = _FakeSkillsBenchEnv()
    sample = Sample(id=0, input='do task', target='', metadata={'task_id': 'task/one'})

    def fake_run_external_agent(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs['environment_override'] is env
        assert kwargs['close_environment'] is False
        assert env.closed is False
        output = ModelOutput(
            model='fake',
            choices=[ChatCompletionChoice.from_content('done')],
            metadata={'source': 'test'},
        )
        return InferenceResult(
            output=output,
            messages=[ChatMessageAssistant(content='done')],
        )

    async def fake_run_verifier(run_env, run_sample):  # type: ignore[no-untyped-def]
        assert run_env is env
        assert env.closed is False
        run_sample.metadata['reward'] = 1.0

    monkeypatch.setattr(adapter, '_build_environment', lambda _: env)
    monkeypatch.setattr('evalscope.agent.external.adapter.run_external_agent', fake_run_external_agent)
    monkeypatch.setattr(adapter, '_run_verifier', fake_run_verifier)

    result = adapter._on_inference(model=None, sample=sample)

    assert result.output.message.text == 'done'
    assert sample.metadata['reward'] == 1.0
    assert env.closed


def test_native_preparation_failure_skips_verifier(monkeypatch) -> None:
    adapter = SkillsBenchAdapter(
        benchmark_meta=BenchmarkMeta(name='skillsbench', dataset_id='skillsbench', metric_list=['score']),
        task_config=TaskConfig(agent_config=NativeAgentConfig(tools=['bash'])),
    )
    env = _FakeSkillsBenchEnv()
    env.prepare = AsyncMock(side_effect=RuntimeError('sandbox unavailable'))
    verifier = AsyncMock(side_effect=AssertionError('verifier must not run'))
    monkeypatch.setattr(adapter, '_build_environment', lambda _: env)
    monkeypatch.setattr(adapter, '_run_verifier', verifier)
    with pytest.raises(RuntimeError, match='sandbox unavailable'):
        adapter._on_inference(model=type('Model', (), {'name': 'mock'})(),
                              sample=Sample(id=0, input='task', target=''))
    verifier.assert_not_awaited()
    assert env.closed


class _FakeSkillsBenchEnv(AgentEnvironment):
    name = 'fake'

    def __init__(self) -> None:
        self.closed = False

    async def exec(self, cmd, *, cwd=None, input=None, timeout=None, env=None):  # type: ignore[no-untyped-def]
        raise AssertionError('exec should not be called in this test')

    async def close(self) -> None:
        self.closed = True


class _TimeoutVerifierEnv(AgentEnvironment):
    name = 'timeout'

    async def exec(self, cmd, *, cwd=None, input=None, timeout=None, env=None):  # type: ignore[no-untyped-def]
        command = ' '.join(str(part) for part in cmd)
        if '/verifier/test.sh' in command:
            return ExecResult(returncode=-1, timed_out=True)
        if 'test-stdout.txt' in command:
            return ExecResult(stdout='partial stdout')
        if 'reward.txt' in command:
            return ExecResult(stdout='1.0\n')
        return ExecResult()

    async def put_dir(self, source_dir: str | Path, target_dir: str) -> None:
        return None

    async def close(self) -> None:
        return None


def _make_task_env(tmp_path: Path) -> Path:
    task = tmp_path / 'task'
    env = task / 'environment'
    skill = env / 'skills' / 'demo'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(
        """---
name: demo
description: Demo skill.
---
""",
        encoding='utf-8',
    )
    (env / 'Dockerfile').write_text(
        """FROM python:3.11-slim
# The official harness may mention /app/skills in comments.
COPY skills /root/.codex/skills
RUN echo ready
""",
        encoding='utf-8',
    )
    return task


def _cleanup(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
