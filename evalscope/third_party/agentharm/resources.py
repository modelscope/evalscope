"""Resolve only the bundled fixtures referenced by the fixed upstream tools."""

import json
from pathlib import Path

_ROOT = Path(__file__).parent
MANIFEST = json.loads((_ROOT / 'manifest.json').read_text(encoding='utf-8'))
DATA_MANIFEST = json.loads((_ROOT / 'data_manifest.json').read_text(encoding='utf-8'))


def resource_path(task_kind: str, upstream_path: str) -> Path:
    """Resolve an upstream fixture literal, including the graders' legacy tools prefix."""
    prefixes = (f'benchmark/{task_kind}_tools/content/', 'benchmark/tools/content/')
    filename = next((upstream_path[len(p) :] for p in prefixes if upstream_path.startswith(p)), '')
    if filename not in MANIFEST[task_kind]['resources']:
        raise ValueError(f'Unknown AgentHarm {task_kind} fixture: {upstream_path!r}')
    namespace = 'common_tools' if filename in MANIFEST['common_resources'] else f'{task_kind}_tools'
    return _ROOT / namespace / 'content' / filename
