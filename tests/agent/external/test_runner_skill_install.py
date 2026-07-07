import asyncio
import pytest
from typing import Dict, List, Optional

from evalscope.agent.external.runners.base import ExternalAgentTask
from evalscope.agent.external.runners.install_helper import install_task_skills
from evalscope.agent.skills import ResolvedSkills, SkillMetadata
from evalscope.api.agent.types import ExecResult


class FakeEnvironment:

    name = 'fake'

    def __init__(self, *, returncode: int = 0, stderr: str = '') -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.commands: List[List[str]] = []
        self.timeouts: List[Optional[float]] = []

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
        self.timeouts.append(timeout)
        return ExecResult(returncode=self.returncode, stderr=self.stderr)


def test_install_task_skills_merges_metadata_and_native_paths() -> None:
    env = FakeEnvironment()
    task = _task_with_skills(install_paths=['$HOME/.agents/skills'])

    asyncio.run(
        install_task_skills(
            env,
            task,
            home_dir='/tmp/eval-home',
            native_install_paths=['$HOME/.claude/skills'],
            runner_name='ClaudeCodeRunner',
        )
    )

    assert len(env.commands) == 1
    assert env.commands[0][:2] == ['bash', '-lc']
    command = env.commands[0][2]
    assert 'mkdir -p /tmp/eval-home/.agents/skills' in command
    assert 'mkdir -p /tmp/eval-home/.claude/skills' in command
    assert 'cp -R /skills/. /tmp/eval-home/.agents/skills/' in command
    assert env.timeouts == [60]


def test_install_task_skills_deduplicates_paths() -> None:
    env = FakeEnvironment()
    task = _task_with_skills(install_paths=['$HOME/.agents/skills'])

    asyncio.run(
        install_task_skills(
            env,
            task,
            home_dir='/tmp/eval-home',
            native_install_paths=['$HOME/.agents/skills'],
            runner_name='CodexRunner',
        )
    )

    command = env.commands[0][2]
    assert command.count('/tmp/eval-home/.agents/skills') == 2


def test_install_task_skills_preserves_home_expansion_when_home_is_inherited() -> None:
    env = FakeEnvironment()
    task = _task_with_skills(install_paths=['$HOME/.agents/skills'])

    asyncio.run(
        install_task_skills(
            env,
            task,
            home_dir=None,
            native_install_paths=['$HOME/.gemini/skills'],
            runner_name='GeminiCliRunner',
        )
    )

    command = env.commands[0][2]
    assert 'mkdir -p "$HOME"/.agents/skills' in command
    assert 'mkdir -p "$HOME"/.gemini/skills' in command


def test_install_task_skills_noops_without_enabled_skills() -> None:
    env = FakeEnvironment()
    task = ExternalAgentTask(instruction='do work', metadata={'agent_skills': ResolvedSkills().model_dump()})

    asyncio.run(
        install_task_skills(
            env,
            task,
            home_dir='/tmp/eval-home',
            native_install_paths=['$HOME/.agents/skills'],
            runner_name='MockAgentRunner',
        )
    )

    assert env.commands == []


def test_install_task_skills_raises_on_copy_failure() -> None:
    env = FakeEnvironment(returncode=1, stderr='copy failed')
    task = _task_with_skills(install_paths=['$HOME/.agents/skills'])

    with pytest.raises(RuntimeError, match='MockAgentRunner failed to install skills: copy failed'):
        asyncio.run(
            install_task_skills(
                env,
                task,
                home_dir=None,
                native_install_paths=[],
                runner_name='MockAgentRunner',
            )
        )


def _task_with_skills(*, install_paths: List[str]) -> ExternalAgentTask:
    skills = ResolvedSkills(
        enabled=True,
        source='sample',
        sandbox_dir='/skills',
        install_paths=install_paths,
        skills=[
            SkillMetadata(
                name='token-writer',
                description='Writes the expected token.',
                path='$HOME/.agents/skills/token-writer/SKILL.md',
            )
        ],
    )
    return ExternalAgentTask(instruction='do work', metadata={'agent_skills': skills.model_dump()})
