"""Helpers for Agent Skills directory compatibility."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field


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
    """Parse simple YAML frontmatter key/value pairs from a skill file."""
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}
    result: Dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or ':' not in line:
            continue
        key, value = line.split(':', 1)
        result[key.strip()] = value.strip().strip('"\'')
    return result


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


__all__ = [
    'ResolvedSkills',
    'SkillMetadata',
    'discover_skills',
    'format_skills_prompt',
    'install_skills_command',
    'parse_frontmatter',
    'quote_path_with_home',
    'skills_from_sample_metadata',
]
