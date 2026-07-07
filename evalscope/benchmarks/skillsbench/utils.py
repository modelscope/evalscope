from __future__ import annotations

import posixpath
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.dataset import Sample

SKILL_MODE_NO_SKILL = 'no-skill'
SKILL_MODE_WITH_SKILL = 'with-skill'
SKILL_MODES = {SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL}
SKILLS_SANDBOX_DIR = '/skills'

_AGENT_SKILL_PATH_MARKERS = (
    '/.codex/skills',
    '/.agents/skills',
    '/.claude/skills',
    '/.gemini/skills',
    '/.goose/skills',
    '/.factory/skills',
    '/.opencode/skill',
    '/.config/opencode/skills',
)


def stage_environment_context(*, task_dir: Path, skill_mode: str) -> str:
    env_dir = task_dir / 'environment'
    if not env_dir.is_dir():
        raise FileNotFoundError(f'SkillsBench environment directory not found: {env_dir}')
    tmp = tempfile.mkdtemp(prefix=f'evalscope-skillsbench-{task_dir.name}-')
    shutil.copytree(env_dir, tmp, dirs_exist_ok=True)
    dockerfile = Path(tmp) / 'Dockerfile'
    if not dockerfile.is_file():
        raise FileNotFoundError(f'SkillsBench Dockerfile not found: {dockerfile}')
    if skill_mode == SKILL_MODE_NO_SKILL:
        shutil.rmtree(Path(tmp) / 'skills', ignore_errors=True)
        shutil.rmtree(Path(tmp) / '_deps' / 'skills', ignore_errors=True)
        _rewrite_dockerfile_without_skill_copies(dockerfile)
        _assert_no_skill_path_residue(dockerfile, include_neutral=True)
    else:
        _rewrite_dockerfile_without_skill_copies(dockerfile)
        _assert_no_skill_path_residue(dockerfile, include_neutral=False)
        content = dockerfile.read_text(encoding='utf-8')
        dockerfile.write_text(
            f'{content.rstrip()}\n\n# SkillsBench skill injection.\nCOPY skills /skills\n', encoding='utf-8'
        )
    return tmp


def read_task_md(path: Path) -> tuple[Dict[str, Any], str]:
    text = path.read_text(encoding='utf-8')
    if not text.startswith('---'):
        return {}, text.strip()
    end = text.find('\n---', 3)
    if end < 0:
        return {}, text.strip()
    frontmatter_text = text[3:end]
    body = text[text.find('\n', end + 1) + 1:].strip()
    return _parse_simple_yaml(frontmatter_text), body


def nested_float(data: Dict[str, Any], keys: List[str]) -> Optional[float]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return optional_float(current)


def optional_float(value: Any) -> Optional[float]:
    if value in (None, ''):
        return None
    return float(value)


def network_enabled(frontmatter: Dict[str, Any]) -> bool:
    mode = (((frontmatter.get('environment') or {}) if isinstance(frontmatter, dict) else {}).get('network_mode'))
    if isinstance(mode, str) and mode.lower() in {'none', 'no-network', 'disabled', 'off'}:
        return False
    return True


def dockerfile_workdir(dockerfile: Path) -> str:
    if not dockerfile.is_file():
        return '/'
    workdir = '/'
    for raw in dockerfile.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        upper = line.upper()
        if upper.startswith('FROM '):
            workdir = '/'
            continue
        if not upper.startswith('WORKDIR '):
            continue
        value = line.split(None, 1)[1].strip()
        parts = shlex.split(value, comments=True)
        if parts:
            workdir = parts[0] if parts[0].startswith('/') else posixpath.normpath(posixpath.join(workdir, parts[0]))
    return workdir


def as_str_list(value: Any) -> List[str]:
    if value is None or value == '':
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def sample_instruction(sample: Sample) -> str:
    if isinstance(sample.input, str):
        return sample.input
    return '\n\n'.join(getattr(message, 'text', '') or '' for message in sample.input)


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def safe_path_part(value: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('.-')
    return text or 'unknown'


def skillsbench_sandbox_config(metadata: Dict[str, Any]) -> Dict[str, Any]:
    allow_network = network_enabled(metadata.get('frontmatter') or {})
    config: Dict[str, Any] = {
        'image': str(metadata['image_tag']),
        'command': 'sleep infinity',
        'working_dir': str(metadata.get('working_dir') or '/'),
        'tools_config': ['shell_executor'],
        'network_enabled': allow_network,
    }
    if allow_network:
        config['extra_hosts'] = {'host.docker.internal': 'host-gateway'}
    return config


async def read_sandbox_text(env: AgentEnvironment, path: str) -> str:
    result = await env.exec(['bash', '-lc', f'cat {sh_quote(path)} 2>/dev/null || true'], timeout=30)
    return result.stdout or ''


def _rewrite_dockerfile_without_skill_copies(dockerfile: Path) -> None:
    kept = []
    for line in dockerfile.read_text(encoding='utf-8').splitlines():
        if _is_skill_copy_line(line):
            continue
        kept.append(line)
    dockerfile.write_text('\n'.join(kept).rstrip() + '\n', encoding='utf-8')


def _is_skill_copy_line(line: str) -> bool:
    return bool(re.search(r'^\s*COPY\s+(_deps/)?skills\b', line))


def _assert_no_skill_path_residue(dockerfile: Path, *, include_neutral: bool) -> None:
    markers = list(_AGENT_SKILL_PATH_MARKERS)
    if include_neutral:
        markers.append('/skills')
    for lineno, line in enumerate(dockerfile.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if any(marker in line for marker in markers):
            raise ValueError(f'Skill path residue in {dockerfile}:{lineno}: {line}')


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith('#') or ':' not in raw:
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        key, value = raw.strip().split(':', 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == '':
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    text = value.strip().strip('"\'')
    if text.lower() in {'true', 'false'}:
        return text.lower() == 'true'
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
