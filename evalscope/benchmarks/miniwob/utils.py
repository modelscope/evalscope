"""Pinned MiniWoB schedule loading and validation helpers."""

from __future__ import annotations

import ast
import csv
import hashlib
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR
from evalscope.utils.download_utils import download_url, file_sha256

BROWSERGYM_VERSION = '0.14.3'
BROWSERGYM_COMMIT = '0a785fbed075224ae81ca9c1fe924f66050696fe'
BROWSERGYM_METADATA_SHA256 = '37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09'
BROWSERGYM_METADATA_URL = (
    f'https://raw.githubusercontent.com/ServiceNow/BrowserGym/{BROWSERGYM_COMMIT}/'
    'browsergym/experiments/src/browsergym/experiments/benchmark/metadata/miniwob.csv'
)
MINIWOB_SCHEDULE_SHA256_BY_REPEATS = {
    1: '7e8487ae966899585f6c1aac78fee869d746cf2c983aae55783356bf5be66926',
    5: '2215888dc6b2cf18bbe2f598d747c21c60d11d27d3b42d030ac2e5622fd865de',
}
MINIWOB_PROFILE_PREFIX = 'openenv_v0.4.1_miniwob_all'
MINIWOB_TASK_COUNT = 125
MINIWOB_REPEATS = 5
MINIWOB_MAX_STEPS = 10
MINIWOB_SEED_MAX = 2**32
MINIWOB_ALL_ACTIONS = (
    'noop',
    'mouse_move',
    'mouse_click',
    'mouse_dblclick',
    'mouse_down',
    'mouse_up',
    'scroll',
    'click',
    'keyboard_press',
    'keyboard_type',
    'fill',
)

_EXPECTED_FIELDS = [
    'task_name',
    'miniwob_category',
    'comment',
    'webgum_subset',
    'similarity_group',
    'browsergym_split',
]


def load_miniwob_records(
    cache_root: Optional[str | Path] = None,
    repeats: int = 1,
) -> tuple[List[Dict[str, Any]], Path]:
    """Download/cache official metadata and attach deterministic episode seeds."""
    if repeats < 1:
        raise ValueError('MiniWoB repeats must be at least 1.')

    metadata_path = ensure_miniwob_metadata(cache_root)
    with metadata_path.open(encoding='utf-8', newline='') as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames != _EXPECTED_FIELDS:
            raise ValueError(
                f'Unexpected MiniWoB metadata fields: {reader.fieldnames!r}; expected {_EXPECTED_FIELDS!r}.'
            )
        rows = list(reader)

    task_names = [row['task_name'] for row in rows]
    if len(rows) != MINIWOB_TASK_COUNT:
        raise ValueError(f'Expected {MINIWOB_TASK_COUNT} MiniWoB tasks, found {len(rows)}.')
    if len(set(task_names)) != len(task_names):
        raise ValueError('MiniWoB metadata contains duplicate task_name values.')
    if task_names != sorted(task_names):
        raise ValueError('MiniWoB metadata task_name values must be ordered.')
    if any(not task_name.startswith('miniwob.') for task_name in task_names):
        raise ValueError("Every MiniWoB task_name must start with 'miniwob.'.")

    rng = np.random.RandomState(42)
    records = []
    for row in rows:
        task_id = row['task_name']
        episode_seeds = [int(seed) for seed in rng.randint(low=0, high=MINIWOB_SEED_MAX, size=repeats, dtype=np.int64)]
        records.append({
            **row,
            'task_id': task_id,
            'openenv_task_name': task_id.removeprefix('miniwob.'),
            '_episode_seeds': episode_seeds,
        })

    schedule_sha256 = hashlib.sha256(
        '\n'.join(f"{record['task_id']}:{seed}" for record in records for seed in record['_episode_seeds']).encode()
    ).hexdigest()
    expected_schedule_sha256 = MINIWOB_SCHEDULE_SHA256_BY_REPEATS.get(repeats)
    if expected_schedule_sha256 is not None and schedule_sha256 != expected_schedule_sha256:
        raise ValueError(
            f'MiniWoB schedule checksum mismatch for repeats={repeats}: '
            f'expected {expected_schedule_sha256}, found {schedule_sha256}.'
        )
    return records, metadata_path


def ensure_miniwob_metadata(cache_root: Optional[str | Path] = None) -> Path:
    """Return a verified cached copy of BrowserGym's MiniWoB metadata CSV."""
    root = Path(cache_root or DEFAULT_EVALSCOPE_CACHE_DIR).expanduser()
    destination = root / 'sources' / 'browsergym' / BROWSERGYM_COMMIT / 'miniwob.csv'
    if destination.is_file() and file_sha256(str(destination)) == BROWSERGYM_METADATA_SHA256:
        return destination

    try:
        download_url(
            BROWSERGYM_METADATA_URL,
            str(destination),
            sha256=BROWSERGYM_METADATA_SHA256,
            headers={'User-Agent': 'EvalScope-MiniWoB/1.0'},
        )
    except Exception as exc:
        raise RuntimeError(
            f'Unable to load pinned MiniWoB metadata. Cache path: {destination}; '
            f'source: {BROWSERGYM_METADATA_URL}. No ModelScope or Hugging Face fallback is configured.'
        ) from exc
    return destination


def validate_browser_action(action: str) -> str:
    """Require one plain function-call expression per browser tool invocation."""
    text = action.strip()
    try:
        tree = ast.parse(text, mode='exec')
    except SyntaxError as exc:
        raise ValueError(f'Browser action is not valid Python-call syntax: {exc.msg}.') from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
        raise ValueError('browser_action requires exactly one function-call expression.')
    call = tree.body[0].value
    if not isinstance(call.func, ast.Name):
        raise ValueError('browser_action requires a direct BrowserGym function name.')
    if call.func.id not in MINIWOB_ALL_ACTIONS:
        raise ValueError(f'Unsupported BrowserGym miniwob_all action: {call.func.id}.')
    if sum(isinstance(node, ast.Call) for node in ast.walk(call)) != 1:
        raise ValueError('browser_action cannot contain nested function calls.')
    if call.func.id == 'click':
        if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise ValueError('click requires a string BID; use mouse_click(x, y) for screenshot coordinates.')
    return text


def miniwob_profile(max_steps: int) -> str:
    """Return the MiniWoB runtime profile name for a step budget."""
    return f'{MINIWOB_PROFILE_PREFIX}_{max_steps}_steps'


__all__ = [
    'BROWSERGYM_COMMIT',
    'BROWSERGYM_METADATA_SHA256',
    'BROWSERGYM_METADATA_URL',
    'BROWSERGYM_VERSION',
    'MINIWOB_ALL_ACTIONS',
    'MINIWOB_MAX_STEPS',
    'MINIWOB_PROFILE_PREFIX',
    'MINIWOB_SCHEDULE_SHA256_BY_REPEATS',
    'ensure_miniwob_metadata',
    'load_miniwob_records',
    'miniwob_profile',
    'validate_browser_action',
]
