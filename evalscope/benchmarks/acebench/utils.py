# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared helpers for the ACEBench adapter: category bookkeeping and record decoding."""
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

# Official category groups, see ACEBench/category.py
ACE_DATA_CATEGORY = {
    'normal': [
        'normal_single_turn_single_function',
        'normal_single_turn_parallel_function',
        'normal_multi_turn_user_adjust',
        'normal_multi_turn_user_switch',
        'normal_similar_api',
        'normal_preference',
        'normal_atom_bool',
        'normal_atom_enum',
        'normal_atom_number',
        'normal_atom_list',
        'normal_atom_object_deep',
        'normal_atom_object_short',
    ],
    'special': [
        'special_incomplete',
        'special_error_param',
        'special_irrelevant',
    ],
    'agent': [
        'agent_multi_step',
        'agent_multi_turn',
    ],
    'atom': [
        'normal_atom_bool',
        'normal_atom_enum',
        'normal_atom_number',
        'normal_atom_list',
        'normal_atom_object_deep',
        'normal_atom_object_short',
    ],
    'multi_turn': [
        'normal_multi_turn_user_adjust',
        'normal_multi_turn_user_switch',
    ],
}

ACEBENCH_SPLITS = ('normal', 'special', 'agent')
ACEBENCH_LANGUAGES = ('en', 'zh')

ACE_DATA_CATEGORY['test_all'] = [category for split in ACEBENCH_SPLITS for category in ACE_DATA_CATEGORY[split]]

# Fine-grained category -> data family, which is also the split it is stored in.
ACEBENCH_CATEGORIES = {category: category.split('_')[0] for category in ACE_DATA_CATEGORY['test_all']}

_TRAILING_INDEX_RE = re.compile(r'(_\d+)+$')


def split_of_category(category: str) -> str:
    """Return the data family (and hub split) a fine-grained category belongs to."""
    return ACEBENCH_CATEGORIES.get(category, category.split('_')[0])


def resolve_categories(names: List[str]) -> List[str]:
    """Expand family names such as ``normal`` or ``atom`` into fine-grained categories.

    Mirrors the official CLI, which accepts both ``--category normal`` and
    ``--category normal_atom_bool``.
    """
    resolved: List[str] = []
    for name in names or []:
        for category in ACE_DATA_CATEGORY.get(name, [name]):
            if category not in resolved:
                resolved.append(category)
    unknown = [category for category in resolved if category not in ACEBENCH_CATEGORIES]
    if unknown:
        raise ValueError(f'Unknown ACEBench categories: {unknown}. Available: {list(ACEBENCH_CATEGORIES)}')
    return resolved


def category_of_record(record: Dict[str, Any]) -> str:
    """Determine the fine-grained category of a record."""
    sub_category = str(record.get('sub_category') or '')
    if sub_category:
        # The hub dataset stores the source file name, e.g. ``data_normal_atom_bool``.
        return sub_category[len('data_'):] if sub_category.startswith('data_') else sub_category
    return _TRAILING_INDEX_RE.sub('', str(record.get('id') or ''))


def dialogue_id_of(record_id: str, test_category: str) -> str:
    """Return the dialogue a record belongs to.

    ``normal_multi_turn_*`` ids are ``<category>_<dialogue>_<step>`` and the official evaluator
    scores those categories per dialogue, so the step index is dropped here.
    """
    if 'multi_turn' in test_category and 'agent' not in test_category:
        return record_id.rsplit('_', 1)[0]
    return record_id


def decode_maybe_json(value: Any, default: Any) -> Any:
    """Decode JSON strings while leaving already decoded values untouched."""
    if value is None or value == '':
        return deepcopy(default)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return deepcopy(default)
    return value


def extract_bracket_blocks(text: Any) -> List[str]:
    """Return every balanced top-level ``[...]`` block from text, in order."""
    if not isinstance(text, str):
        return []

    blocks: List[str] = []
    start = -1
    depth = 0
    for index, char in enumerate(text):
        if char == '[':
            if depth == 0:
                start = index
            depth += 1
        elif char == ']' and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start:index + 1])
                start = -1
    return blocks


def extract_outermost_bracket_content(text: str) -> Optional[str]:
    """Return the first balanced ``[...]`` block from text."""
    blocks = extract_bracket_blocks(text)
    return blocks[0] if blocks else None
