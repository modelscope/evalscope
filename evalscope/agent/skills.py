"""Helpers for Agent Skills directory compatibility."""

from __future__ import annotations

import re
import shlex
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

if TYPE_CHECKING:
    from evalscope.api.agent import AgentEnvironment

CONFIG_SKILL_SOURCE = 'config'
TASK_BUNDLED_SKILL_SOURCE = 'task_bundled'
DEFAULT_SKILLS_SANDBOX_DIR = '/tmp/evalscope-agent-skills'
DEFAULT_SKILLS_INSTALL_DIR = '$HOME/.agents/skills'


class SkillMetadata(BaseModel):
    """Metadata discovered from a ``SKILL.md`` file."""

    name: str
    description: str = ''
    path: str


class ResolvedSkills(BaseModel):
    """Resolved skills for one agent run."""

    enabled: bool = False
    source: str = 'none'
    host_dir: str | None = None
    sandbox_dir: str | None = None
    prompt_base_dir: str | None = None
    install_paths: List[str] = Field(default_factory=list)
    skills: List[SkillMetadata] = Field(default_factory=list)
    metadata_errors: List[str] = Field(default_factory=list)


def discover_skills(skills_dir: str | Path, *, path_prefix: str | None = None) -> tuple[List[SkillMetadata], List[str]]:
    """Discover immediate child skill directories containing ``SKILL.md``."""
    base = Path(skills_dir)
    skills: List[SkillMetadata] = []
    errors: List[str] = []
    if not base.is_dir():
        return skills, [f'skills_dir is not a directory: {base}']

    for skill_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        skill_file = skill_dir / 'SKILL.md'
        if not skill_file.is_file():
            continue
        try:
            content = skill_file.read_text(encoding='utf-8')
        except OSError as exc:
            errors.append(f'{skill_file}: {exc}')
            continue
        frontmatter = parse_frontmatter(content)
        name = frontmatter.get('name') or skill_dir.name
        description = frontmatter.get('description') or ''
        if not frontmatter.get('name') or not frontmatter.get('description'):
            errors.append(f'{skill_file}: missing name or description frontmatter')
        display_base = (path_prefix.rstrip('/') if path_prefix else str(skill_dir.parent))
        skills.append(
            SkillMetadata(
                name=name,
                description=description,
                path=f'{display_base}/{skill_dir.name}/SKILL.md',
            )
        )
    return skills, errors


def parse_frontmatter(content: str) -> Dict[str, str]:
    """Parse YAML frontmatter key/value pairs from a skill file."""
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}
    parsed = yaml.safe_load(match.group(1)) or {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): '' if value is None else str(value) for key, value in parsed.items()}


def format_skills_prompt(skills: List[SkillMetadata]) -> str:
    """Render a neutral prompt nudge for available skills."""
    if not skills:
        return ''
    lines = [
        'You have access to the following skills. Each skill is a directory containing a SKILL.md file with '
        'instructions and resources. When a skill is relevant, read its SKILL.md before using it.',
        '',
    ]
    for skill in skills:
        description = f': {skill.description}' if skill.description else ''
        lines.append(f'- {skill.name}{description} ({skill.path})')
    return '\n'.join(lines)


def skills_from_sample_metadata(metadata: Dict[str, Any]) -> ResolvedSkills:
    """Build ``ResolvedSkills`` from sample metadata."""
    raw = metadata.get('agent_skills') or {}
    if isinstance(raw, ResolvedSkills):
        return raw
    if isinstance(raw, dict):
        try:
            return ResolvedSkills.model_validate(raw)
        except Exception:
            return ResolvedSkills()
    return ResolvedSkills()


def resolve_agent_skills(
    *,
    sample_metadata: Dict[str, Any],
    config_skills_dir: str | None,
    prompt_base_dir: str = DEFAULT_SKILLS_INSTALL_DIR,
    install_paths: Iterable[str] = (DEFAULT_SKILLS_INSTALL_DIR, ),
    sandbox_dir: str = DEFAULT_SKILLS_SANDBOX_DIR,
) -> ResolvedSkills:
    """Resolve sample-bundled skills first, then user-configured skills."""
    sample_skills = skills_from_sample_metadata(sample_metadata)
    if sample_skills.enabled:
        return sample_skills
    if not config_skills_dir:
        return ResolvedSkills()

    host_dir = Path(config_skills_dir).expanduser()
    if not host_dir.is_dir():
        raise FileNotFoundError(f'skills_dir is not a directory: {host_dir}')

    skills, errors = discover_skills(host_dir, path_prefix=prompt_base_dir)
    return ResolvedSkills(
        enabled=bool(skills),
        source=CONFIG_SKILL_SOURCE,
        host_dir=str(host_dir),
        sandbox_dir=sandbox_dir,
        prompt_base_dir=prompt_base_dir,
        install_paths=list(install_paths),
        skills=skills,
        metadata_errors=errors,
    )


async def install_agent_skills(
    environment: 'AgentEnvironment',
    skills: ResolvedSkills,
    *,
    install_paths: Iterable[str] | None = None,
    runner_name: str,
    timeout: float = 60,
) -> None:
    """Stage and install resolved skills into paths visible to an agent."""
    if not skills.enabled:
        return
    await stage_agent_skills(environment, skills, runner_name=runner_name)

    resolved_install_paths = _dedupe_paths(list(install_paths) if install_paths is not None else skills.install_paths)
    command = install_skills_command(skills.sandbox_dir or '', resolved_install_paths)
    if not command:
        return
    result = await environment.exec(['bash', '-lc', command], timeout=timeout)
    if result.returncode != 0:
        detail = ((result.stderr or result.stdout or '').strip() or f'rc={result.returncode}')[-1000:]
        raise RuntimeError(f'{runner_name} failed to install skills: {detail}')


async def stage_agent_skills(environment: 'AgentEnvironment', skills: ResolvedSkills, *, runner_name: str) -> None:
    """Upload host-configured skills into the environment when needed."""
    if not skills.enabled or skills.source != CONFIG_SKILL_SOURCE:
        return
    if not skills.host_dir or not skills.sandbox_dir:
        raise RuntimeError(f'{runner_name} received config skills without host_dir and sandbox_dir')
    try:
        await environment.put_dir(skills.host_dir, skills.sandbox_dir)
    except NotImplementedError as exc:
        raise RuntimeError(f'{runner_name} requires environment.put_dir to install skills_dir') from exc
    except Exception as exc:
        raise RuntimeError(f'{runner_name} failed to stage skills: {exc}') from exc


def install_skills_command(source_dir: str, install_paths: List[str]) -> str | None:
    """Return a POSIX shell command copying skills into discovery paths."""
    if not source_dir or not install_paths:
        return None
    commands = []
    quoted_source = shlex.quote(source_dir.rstrip('/'))
    for dest in install_paths:
        quoted_dest = quote_path_with_home(dest.rstrip('/'))
        commands.append(f'mkdir -p {quoted_dest} && cp -R {quoted_source}/. {quoted_dest}/')
    return ' && '.join(commands)


def quote_path_with_home(path: str) -> str:
    """Quote a shell path while preserving leading ``$HOME`` expansion."""
    if path == '$HOME':
        return '"$HOME"'
    if path.startswith('$HOME/'):
        rest = path[len('$HOME/'):]
        if not rest:
            return '"$HOME"'
        return f'"$HOME"/{shlex.quote(rest)}'
    return shlex.quote(path)


def _dedupe_paths(paths: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


__all__ = [
    'CONFIG_SKILL_SOURCE',
    'DEFAULT_SKILLS_INSTALL_DIR',
    'DEFAULT_SKILLS_SANDBOX_DIR',
    'ResolvedSkills',
    'SkillMetadata',
    'TASK_BUNDLED_SKILL_SOURCE',
    'discover_skills',
    'format_skills_prompt',
    'install_agent_skills',
    'install_skills_command',
    'parse_frontmatter',
    'quote_path_with_home',
    'resolve_agent_skills',
    'stage_agent_skills',
    'skills_from_sample_metadata',
]
