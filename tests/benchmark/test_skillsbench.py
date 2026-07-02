from __future__ import annotations

import subprocess
from pathlib import Path

from evalscope.agent.skills import discover_skills, format_skills_prompt, install_skills_command
from evalscope.api.agent.types import NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.sandbox.docker_image import DockerImageSpec, hash_build_context
from evalscope.benchmarks.bigcodebench.bigcodebench_adapter import BigCodeBenchAdapter
from evalscope.benchmarks.skillsbench.skillsbench_adapter import (
    _SKILL_MODE_NO_SKILL,
    _SKILL_MODE_WITH_SKILL,
    SkillsBenchAdapter,
    _read_task_md,
    _stage_environment_context,
    _wrap_input_command,
    _wrap_timeout_command,
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

    frontmatter, body = _read_task_md(task_md)

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

    context = Path(_stage_environment_context(task_dir=task, skill_mode=_SKILL_MODE_NO_SKILL))

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

    context = Path(_stage_environment_context(task_dir=task, skill_mode=_SKILL_MODE_WITH_SKILL))

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


def test_wrap_timeout_command_quotes_arguments() -> None:
    cmd = _wrap_timeout_command(['bash', '-lc', "echo 'hello world'"], 2.5)

    assert cmd[0:2] == ['bash', '-lc']
    assert 'timeout --kill-after=5s 2s' in cmd[2]
    assert "'echo '\"'\"'hello world'\"'\"''" in cmd[2]


def test_wrap_input_command_pipes_stdin() -> None:
    cmd = _wrap_input_command(['python', '-c', 'import sys; print(sys.stdin.read())'], "hello 'world'")

    assert cmd[0:2] == ['bash', '-lc']
    assert "printf %s 'hello '\"'\"'world'\"'\"''" in cmd[2]
    assert "'import sys; print(sys.stdin.read())'" in cmd[2]


def test_agent_config_accepts_skills_fields() -> None:
    cfg = NativeAgentConfig(skills_dir='/tmp/skills', skill_prompt_nudge=False)

    assert cfg.skills_dir == '/tmp/skills'
    assert cfg.skill_prompt_nudge is False


def test_bigcodebench_optional_builder_overrides_sandbox_image(monkeypatch, tmp_path: Path) -> None:
    from evalscope.api.sandbox.docker_image import DockerImageResult

    def fake_build_or_reuse(self, spec):  # type: ignore[no-untyped-def]
        assert spec.context_dir == str(tmp_path)
        assert spec.cache_key_parts == ['bigcodebench', 'bigcodebench']
        return DockerImageResult(image_tag='evalscope-bigcodebench:test', reused=True, context_hash='abc')

    monkeypatch.setattr('evalscope.benchmarks.bigcodebench.bigcodebench_adapter.DockerImageBuilder.build_or_reuse',
                        fake_build_or_reuse)
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

    assert adapter._benchmark_meta.sandbox_config['image'] == 'evalscope-bigcodebench:test'


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
