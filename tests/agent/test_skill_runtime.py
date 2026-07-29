import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from evalscope.agent.environments.local import LocalAgentEnvironment
from evalscope.agent.skills import (
    DEFAULT_SKILLS_INSTALL_DIR,
    DEFAULT_SKILLS_SANDBOX_DIR,
    ResolvedSkills,
    SkillMetadata,
    install_agent_skills,
    resolve_agent_skills,
)
from evalscope.api.agent.types import ExecResult


class FakeEnvironment:

    def __init__(self) -> None:
        self.put_dirs: List[tuple[str, str]] = []
        self.commands: List[List[str]] = []

    async def put_dir(self, source_dir: str | Path, target_dir: str) -> None:
        self.put_dirs.append((str(source_dir), target_dir))

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        self.commands.append(cmd)
        return ExecResult()


def test_resolve_config_skills_discovers_and_uses_sandbox_stage_dir(tmp_path: Path) -> None:
    skill_dir = tmp_path / 'skills' / 'demo'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text(
        """---
name: demo
description: Demo skill.
---
""",
        encoding='utf-8',
    )

    skills = resolve_agent_skills(sample_metadata={}, config_skills_dir=str(tmp_path / 'skills'))

    assert skills.enabled
    assert skills.source == 'config'
    assert skills.host_dir == str(tmp_path / 'skills')
    assert skills.sandbox_dir == DEFAULT_SKILLS_SANDBOX_DIR
    assert skills.install_paths == [DEFAULT_SKILLS_INSTALL_DIR]
    assert skills.skills[0].path == '$HOME/.agents/skills/demo/SKILL.md'


def test_install_config_skills_uploads_then_installs(tmp_path: Path) -> None:
    skill_root = tmp_path / 'skills'
    skill_root.mkdir()
    env = FakeEnvironment()
    skills = ResolvedSkills(
        enabled=True,
        source='config',
        host_dir=str(skill_root),
        sandbox_dir=DEFAULT_SKILLS_SANDBOX_DIR,
        install_paths=[DEFAULT_SKILLS_INSTALL_DIR],
        skills=[SkillMetadata(name='demo', description='Demo', path='$HOME/.agents/skills/demo/SKILL.md')],
    )

    asyncio.run(install_agent_skills(env, skills, runner_name='NativeAgentRunner'))

    assert env.put_dirs == [(str(skill_root), DEFAULT_SKILLS_SANDBOX_DIR)]
    assert len(env.commands) == 1
    command = env.commands[0][2]
    assert f'cp -R {DEFAULT_SKILLS_SANDBOX_DIR}/.' in command
    assert '"$HOME"/.agents/skills' in command


def test_install_task_bundled_skills_does_not_upload() -> None:
    env = FakeEnvironment()
    skills = ResolvedSkills(
        enabled=True,
        source='task_bundled',
        sandbox_dir='/skills',
        install_paths=[DEFAULT_SKILLS_INSTALL_DIR],
        skills=[SkillMetadata(name='demo', description='Demo', path='$HOME/.agents/skills/demo/SKILL.md')],
    )

    asyncio.run(install_agent_skills(env, skills, runner_name='CodexRunner'))

    assert env.put_dirs == []
    assert len(env.commands) == 1
    assert 'cp -R /skills/. "$HOME"/.agents/skills/' in env.commands[0][2]


def test_local_environment_put_dir_copies_directory(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    source.mkdir()
    (source / 'SKILL.md').write_text('skill', encoding='utf-8')

    asyncio.run(LocalAgentEnvironment().put_dir(source, str(target)))

    assert (target / 'SKILL.md').read_text(encoding='utf-8') == 'skill'
