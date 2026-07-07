from __future__ import annotations

from pathlib import Path

from evalscope.agent.external.config import ExternalAgentConfig
from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult, NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.benchmarks.skillsbench.skillsbench_adapter import SkillsBenchAdapter
from evalscope.benchmarks.skillsbench.utils import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_WITH_SKILL,
    skillsbench_sandbox_config,
    stage_environment_context,
)
from evalscope.config import TaskConfig


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


def test_run_verifier_records_timeout_before_reward_parsing(tmp_path: Path) -> None:
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
    adapter._current_output_dir = str(tmp_path)
    sample = Sample(
        id=0,
        input='do task',
        target='',
        metadata={
            'task_id': 'task/one',
            'task_dir': str(task),
            'skill_mode': 'with-skill',
            'verifier_timeout_sec': 1,
        },
    )
    env = _TimeoutVerifierEnv()

    import asyncio

    asyncio.run(adapter._run_verifier(env, sample))

    assert sample.metadata['verifier_timed_out'] is True
    assert sample.metadata['verifier_error'] == 'Verifier timed out'
    assert sample.metadata['reward'] == 0.0


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
