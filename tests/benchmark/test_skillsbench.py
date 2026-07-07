from __future__ import annotations

import subprocess
from pathlib import Path

from evalscope.agent.external.config import ExternalAgentConfig
from evalscope.agent.skills import discover_skills, format_skills_prompt, install_skills_command
from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.sandbox.docker_image import DockerImageSpec, hash_build_context
from evalscope.benchmarks.bigcodebench.bigcodebench_adapter import BigCodeBenchAdapter
from evalscope.benchmarks.skillsbench.skillsbench_adapter import SkillsBenchAdapter
from evalscope.benchmarks.skillsbench.utils import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_WITH_SKILL,
    dockerfile_workdir,
    network_enabled,
    read_task_md,
    skillsbench_sandbox_config,
    stage_environment_context,
)
from evalscope.config import TaskConfig


def test_read_task_md_removes_frontmatter(tmp_path: Path) -> None:
    task_md = tmp_path / 'task.md'
    task_md.write_text(
        """---
schema_version: '1.3'
agent:
  timeout_sec: 900.0
---

Do the task.
""",
        encoding='utf-8',
    )

    frontmatter, body = read_task_md(task_md)

    assert frontmatter['agent']['timeout_sec'] == 900.0
    assert body == 'Do the task.'


def test_discover_skills_and_prompt(tmp_path: Path) -> None:
    skill_dir = tmp_path / 'skills' / 'office'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text(
        """---
name: office-docs
description: Work with office documents.
---

Use python-docx.
""",
        encoding='utf-8',
    )

    skills, errors = discover_skills(tmp_path / 'skills', path_prefix='$HOME/.agents/skills')
    prompt = format_skills_prompt(skills)

    assert errors == []
    assert [skill.name for skill in skills] == ['office-docs']
    assert '$HOME/.agents/skills/office/SKILL.md' in prompt
    assert 'Work with office documents.' in prompt


def test_install_skills_command_preserves_home_and_fails_on_copy_error(tmp_path: Path) -> None:
    command = install_skills_command('/missing-skills', ['$HOME/.agents/skills'])

    assert command is not None
    assert '"$HOME"/.agents/skills' in command
    assert '|| true' not in command
    assert '2>/dev/null' not in command
    result = subprocess.run(['bash', '-lc', command], env={'HOME': str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode != 0


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


def test_stage_with_skill_injects_neutral_skills_path(tmp_path: Path) -> None:
    task = _make_task_env(tmp_path)

    context = Path(stage_environment_context(task_dir=task, skill_mode=SKILL_MODE_WITH_SKILL))

    try:
        dockerfile = (context / 'Dockerfile').read_text(encoding='utf-8')
        assert 'COPY skills /skills' in dockerfile
        assert '/root/.codex/skills' not in dockerfile
    finally:
        _cleanup(context)


def test_hash_build_context_includes_cache_key_parts(tmp_path: Path) -> None:
    (tmp_path / 'Dockerfile').write_text('FROM scratch\n', encoding='utf-8')

    first = hash_build_context(str(tmp_path), cache_key_parts=['no-skill'])
    second = hash_build_context(str(tmp_path), cache_key_parts=['with-skill'])

    assert first != second


def test_docker_builder_hash_includes_build_args(monkeypatch, tmp_path: Path) -> None:
    from evalscope.api.sandbox.docker_image import DockerImageBuilder

    (tmp_path / 'Dockerfile').write_text('FROM scratch\n', encoding='utf-8')
    monkeypatch.setattr('evalscope.api.sandbox.docker_image.should_build_docker_image', lambda image: False)

    first = DockerImageBuilder().build_or_reuse(
        DockerImageSpec(name_prefix='demo', context_dir=str(tmp_path), build_args={'A': '1'})
    )
    second = DockerImageBuilder().build_or_reuse(
        DockerImageSpec(name_prefix='demo', context_dir=str(tmp_path), build_args={'A': '2'})
    )

    assert first.image_tag != second.image_tag


def test_network_enabled_honors_no_network_mode() -> None:
    assert network_enabled({'environment': {'network_mode': 'public'}}) is True
    assert network_enabled({'environment': {'network_mode': 'no-network'}}) is False


def test_dockerfile_workdir_uses_final_stage_workdir(tmp_path: Path) -> None:
    dockerfile = tmp_path / 'Dockerfile'
    dockerfile.write_text(
        """FROM python:3.11-slim
WORKDIR /root
RUN echo ready
WORKDIR output
FROM ubuntu:24.04
WORKDIR "/app path" # final
""",
        encoding='utf-8',
    )

    assert dockerfile_workdir(dockerfile) == '/app path'


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


def test_agent_config_accepts_skills_fields() -> None:
    cfg = NativeAgentConfig(skills_dir='/tmp/skills', skill_prompt_nudge=False)

    assert cfg.skills_dir == '/tmp/skills'
    assert cfg.skill_prompt_nudge is False


def test_bigcodebench_optional_builder_returns_sandbox_image_spec(tmp_path: Path) -> None:
    meta = BenchmarkMeta(
        name='bigcodebench',
        dataset_id='evalscope/bigcodebench',
        subset_list=['default'],
        metric_list=['acc'],
        prompt_template='{prompt}',
        sandbox_config={'image': 'bigcodebench/bigcodebench-evaluate:latest'},
        extra_params={
            'split': 'instruct',
            'version': 'default',
            'calibrate': True,
            'docker_build_context': str(tmp_path),
            'dockerfile': 'Dockerfile',
            'force_rebuild': False,
        },
    )
    cfg = TaskConfig(datasets=['bigcodebench'])

    adapter = BigCodeBenchAdapter(benchmark_meta=meta, task_config=cfg)
    spec = adapter.get_sandbox_image_spec()

    assert spec is not None
    assert spec.name_prefix == 'evalscope-bigcodebench'
    assert spec.context_dir == str(tmp_path)
    assert spec.dockerfile == 'Dockerfile'
    assert spec.cache_key_parts == ['bigcodebench', 'bigcodebench']
    assert spec.force_rebuild is False


def test_bigcodebench_default_uses_prebuilt_sandbox_image() -> None:
    meta = BenchmarkMeta(
        name='bigcodebench',
        dataset_id='evalscope/bigcodebench',
        subset_list=['default'],
        metric_list=['acc'],
        prompt_template='{prompt}',
        sandbox_config={'image': 'bigcodebench/bigcodebench-evaluate:latest'},
        extra_params={
            'split': 'instruct',
            'version': 'default',
            'calibrate': True,
            'docker_build_context': '',
            'dockerfile': 'Dockerfile',
            'force_rebuild': False,
        },
    )
    cfg = TaskConfig(datasets=['bigcodebench'])

    adapter = BigCodeBenchAdapter(benchmark_meta=meta, task_config=cfg)

    assert adapter.get_sandbox_image_spec() is None
    assert adapter._benchmark_meta.sandbox_config['image'] == 'bigcodebench/bigcodebench-evaluate:latest'


def test_save_verifier_artifacts_writes_paths(tmp_path: Path) -> None:
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
        metadata={'task_id': 'task/one', 'skill_mode': 'with-skill'},
    )

    adapter._save_verifier_artifacts(sample, stdout='stdout', reward_text='1.0\n', ctrf_text='{"ok": true}')

    artifact_dir = tmp_path / 'artifacts' / 'skillsbench' / 'task-one' / 'with-skill'
    assert (artifact_dir / 'test-stdout.txt').read_text(encoding='utf-8') == 'stdout'
    assert (artifact_dir / 'reward.txt').read_text(encoding='utf-8') == '1.0\n'
    assert (artifact_dir / 'ctrf.json').read_text(encoding='utf-8') == '{"ok": true}'
    assert sample.metadata['artifact_dir'] == str(artifact_dir)
    assert sample.metadata['verifier_stdout_path'] == str(artifact_dir / 'test-stdout.txt')


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
