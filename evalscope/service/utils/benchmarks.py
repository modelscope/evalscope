# Copyright (c) Alibaba, Inc. and its affiliates.
"""Benchmark catalogue and description helpers for the EvalScope service."""
import re
from typing import Any, Dict, List, Optional

from evalscope.utils.resource_utils import load_benchmark_data

# ---------------------------------------------------------------------------
# Supported benchmark catalogue
# ---------------------------------------------------------------------------

#: Default text-only benchmarks exposed by the /api/v1/eval/benchmarks endpoint.
DEFAULT_TEXT_BENCHMARKS: List[str] = [
    'gsm8k',
    'mmlu',
    'cmmlu',
    'ceval',
    'arc',
    'math_500',
    'aime24',
    'aime25',
    'aime26',
    'gpqa_diamond',
    'super_gpqa',
    'commonsense_qa',
    'piqa',
    'siqa',
    'logi_qa',
    'qasc',
    'sciq',
    'race',
    'mgsm',
    'multi_if',
    'ifeval',
    'ifbench',
]

#: Default multimodal benchmarks exposed by the /api/v1/eval/benchmarks endpoint.
DEFAULT_MULTIMODAL_BENCHMARKS: List[str] = [
    'mmmu',
    'cmmmu',
    'mmmu_pro',
    'mm_bench',
    'chartqa',
    'ai2d',
    'real_world_qa',
    'math_vista',
    'math_vision',
    'seed_bench_2_plus',
]

# ---------------------------------------------------------------------------
# Description parsing
# ---------------------------------------------------------------------------

# Mapping of Chinese / English H2 section titles to canonical snake_case keys.
_SECTION_TITLE_TO_KEY: Dict[str, str] = {
    # Chinese → snake_case key
    '概述': 'overview',
    '任务描述': 'task_description',
    '主要特点': 'key_features',
    '主要特性': 'key_features',
    '核心特点': 'key_features',
    '核心特性': 'key_features',
    '评估说明': 'evaluation_notes',
    '评测说明': 'evaluation_notes',
    '属性': 'properties',
    '数据统计': 'data_statistics',
    '样例示例': 'sample_example',
    '提示模板': 'prompt_template',
    '沙箱配置': 'sandbox_configuration',
    '额外参数': 'extra_parameters',
    '使用方法': 'usage',
    # English → snake_case key
    'Overview': 'overview',
    'Task Description': 'task_description',
    'Key Features': 'key_features',
    'Evaluation Notes': 'evaluation_notes',
    'Properties': 'properties',
    'Data Statistics': 'data_statistics',
    'Sample Example': 'sample_example',
    'Prompt Template': 'prompt_template',
    'Sandbox Configuration': 'sandbox_configuration',
    'Extra Parameters': 'extra_parameters',
    'Usage': 'usage',
    # Puzzle-specific headings that appear in some benchmarks
    'Clues for the Example Puzzle': 'clues_for_the_example_puzzle',
    'Answer to the Example Puzzle': 'answer_to_the_example_puzzle',
}


def _normalise_section_key(title: str) -> str:
    """Return the canonical snake_case key for a section heading.

    Falls back to the original title when no mapping is found so that
    unknown headings are still surfaced rather than silently dropped.
    """
    return _SECTION_TITLE_TO_KEY.get(title, title)


def parse_benchmark_description(readme_content: str) -> Dict[str, Any]:
    """Process a README markdown string into structured description data.

    Steps:
    1. Strip the leading H1 heading.
    2. Remove the last H2 section (e.g. ``使用方法`` / ``Usage``).
    3. Split the remaining content by H2 headings into a canonical-English-
       keyed dict (insertion order matches document order).
    4. Return both the full processed text and the per-section dict.

    Args:
        readme_content: Raw markdown from ``readme.zh`` or ``readme.en``.

    Returns:
        A dict with keys:

        - ``full``     – the complete processed markdown string.
        - ``sections`` – an ordered dict mapping each H2 title to its content
          using canonical English keys, e.g.
          ``{"Overview": "...", "Task Description": "...", ...}``.
    """
    lines = readme_content.splitlines()

    # 1. Remove the first H1 heading line
    for i, line in enumerate(lines):
        if re.match(r'^#\s', line):
            lines.pop(i)
            break
        elif line.strip():  # Non-blank, non-H1 line encountered first – stop
            break

    # 2. Remove the last H2 section (heading + everything after it)
    last_h2: Optional[int] = None
    for i, line in enumerate(lines):
        if re.match(r'^##\s', line):
            last_h2 = i
    if last_h2 is not None:
        lines = lines[:last_h2]

    full_content = '\n'.join(lines).strip()

    # 3. Split by H2 headings into an ordered canonical-key → content dict
    sections: Dict[str, str] = {}
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        if re.match(r'^##\s', line):
            if current_title is not None:
                sections[current_title] = '\n'.join(current_lines).strip()
            raw_title = line.lstrip('#').strip()
            current_title = _normalise_section_key(raw_title)
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = '\n'.join(current_lines).strip()

    return {'full': full_content, 'sections': sections}


def build_benchmark_entry(name: str) -> Dict[str, Any]:
    """Load metadata for a single benchmark and build the API response entry.

    Args:
        name: Benchmark identifier (e.g. ``'gsm8k'``).

    Returns:
        A dict containing ``name``, ``meta``, and ``description``.
        ``meta`` includes all top-level benchmark data fields except
        ``readme`` (e.g. ``meta``, ``statistics``, ``sample_example``,
        ``updated_at``).  The ``description`` field has two sub-keys,
        ``'zh'`` and ``'en'``, each containing ``full`` and ``sections``
        parsed from the corresponding readme.  A key is ``None`` when the
        readme for that language is absent.
    """
    data = load_benchmark_data(name).get(name, {})
    readme = data.get('readme', {})

    zh_raw = readme.get('zh') or ''
    en_raw = readme.get('en') or ''

    description = {
        'zh': parse_benchmark_description(zh_raw) if zh_raw else None,
        'en': parse_benchmark_description(en_raw) if en_raw else None,
    }

    meta = data.get('meta', {})

    # Flatten metrics to list[str]: extract key name if item is a dict
    raw_metrics = meta.get('metrics', [])
    metrics = [next(iter(m)) if isinstance(m, dict) else str(m) for m in raw_metrics]

    return {
        'name': name,
        'metrics': metrics,
        'meta': meta,
        'description': description,
    }
