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
    'infovqa',
    'docvqa',
    'math_vista',
    'math_vision',
    'seed_bench_2_plus',
]

# ---------------------------------------------------------------------------
# Description parsing
# ---------------------------------------------------------------------------


def parse_benchmark_description(readme_content: str) -> Dict[str, Any]:
    """Process a README markdown string into structured description data.

    Steps:
    1. Strip the leading H1 heading.
    2. Remove the last H2 section (e.g. ``使用方法`` / ``Usage``).
    3. Split the remaining content by H2 headings into a title-keyed dict
       (insertion order matches document order).
    4. Return both the full processed text and the per-section dict.

    Args:
        readme_content: Raw markdown from ``readme.zh`` (or ``readme.en``).

    Returns:
        A dict with keys:

        - ``full``     – the complete processed markdown string.
        - ``sections`` – an ordered dict mapping each H2 title to its content,
          e.g. ``{"概述": "...", "任务描述": "...", ...}``.
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

    # 3. Split by H2 headings into an ordered title → content dict
    sections: Dict[str, str] = {}
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        if re.match(r'^##\s', line):
            if current_title is not None:
                sections[current_title] = '\n'.join(current_lines).strip()
            current_title = line.lstrip('#').strip()
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
        A dict containing ``name``, ``pretty_name``, ``tags``, and
        ``description`` (with ``full`` and ``sections`` sub-fields).
    """
    data = load_benchmark_data(name).get(name, {})
    meta = data.get('meta', {})
    readme = data.get('readme', {})

    # Prefer Chinese readme; fall back to English; then empty string
    raw_readme = readme.get('zh') or readme.get('en') or ''
    description = parse_benchmark_description(raw_readme)

    return {
        'name': name,
        'pretty_name': meta.get('pretty_name', name),
        'tags': meta.get('tags', []),
        'description': description,
    }
