# Copyright (c) Alibaba, Inc. and its affiliates.
"""Documentation utilities for benchmark data management.

This module provides utilities for:
- Unified benchmark_data.json management
- Content hash for detecting changes and translation updates
- README generation and translation support
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from evalscope.utils.function_utils import thread_safe

# Path to the unified benchmark data file
DOCS_DIR = Path(__file__).parent.parent.parent.parent / 'docs'
BENCHMARK_DATA_JSON_PATH = str(DOCS_DIR / 'asset' / 'source' / 'benchmark_data.json')

# Output directories for benchmark READMEs
BENCHMARK_README_DIR_ZH = DOCS_DIR / 'zh' / 'benchmarks'
BENCHMARK_README_DIR_EN = DOCS_DIR / 'en' / 'benchmarks'
INDEX_DIR_ZH = DOCS_DIR / 'zh' / 'get_started' / 'supported_dataset'
INDEX_DIR_EN = DOCS_DIR / 'en' / 'get_started' / 'supported_dataset'


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash of content for change detection."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def load_benchmark_data(path: str = BENCHMARK_DATA_JSON_PATH) -> Dict[str, Any]:
    """
    Load unified benchmark data from JSON file.

    The benchmark data structure is:
    {
        "benchmark_name": {
            "meta": {  # Basic metadata
                "pretty_name": str,
                "dataset_id": str,
                "paper_url": str,
                "tags": List[str],
                "metrics": List[str],
                "few_shot_num": int,
                "eval_split": str,
                "subset_list": List[str],
                "description": str,
                "prompt_template": str,
                "system_prompt": str,
            },
            "statistics": {  # Data statistics
                "total_samples": int,
                "subset_stats": List[...],
                "prompt_length": {...},
            },
            "sample_example": {  # Representative sample
                "data": {...},
                "subset": str,
                "truncated": bool,
            },
            "readme": {  # README content
                "en": str,  # English README content
                "zh": str,  # Chinese README content (translated)
                "content_hash": str,  # Hash of English content for change detection
                "needs_translation": bool,  # Flag indicating translation needed
            },
            "updated_at": str,  # ISO timestamp
        }
    }
    """
    if not Path(path).exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


@thread_safe
def save_benchmark_data(data: Dict[str, Any], path: str = BENCHMARK_DATA_JSON_PATH):
    """Save unified benchmark data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_benchmark_entry(benchmark_name: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """Get entry for a specific benchmark, creating empty structure if not exists."""
    if data is None:
        data = load_benchmark_data()

    if benchmark_name not in data:
        data[benchmark_name] = {
            'meta': {},
            'statistics': {},
            'sample_example': {},
            'readme': {
                'en': '',
                'zh': '',
                'content_hash': '',
                'needs_translation': False,
            },
            'updated_at': '',
        }
    return data[benchmark_name]


def needs_translation_update(benchmark_name: str, current_en_content: str = None, data: Optional[Dict] = None) -> bool:
    """
    Check if translation needs update by comparing content hash.

    Args:
        benchmark_name: Name of the benchmark
        current_en_content: Current English README content (optional, uses stored if None)
        data: Optional pre-loaded benchmark data

    Returns:
        True if translation needs update (content changed or no translation exists)
    """
    if data is None:
        data = load_benchmark_data()

    entry = data.get(benchmark_name, {})
    readme = entry.get('readme', {})

    # If explicitly marked as needing translation
    if readme.get('needs_translation', False):
        return True

    # If current content provided, check hash
    if current_en_content:
        stored_hash = readme.get('content_hash', '')
        current_hash = compute_content_hash(current_en_content)
        if not stored_hash or stored_hash != current_hash:
            return True

    # If no Chinese translation exists
    if not readme.get('zh', '').strip():
        return True

    return False


def get_benchmarks_needing_translation(data: Optional[Dict] = None) -> list:
    """Get list of benchmark names that need translation update."""
    if data is None:
        data = load_benchmark_data()

    result = []
    for name, entry in data.items():
        readme = entry.get('readme', {})
        en_content = readme.get('en', '')
        if en_content and needs_translation_update(name, en_content, data):
            result.append(name)
    return result


# Export main functions
__all__ = [
    'BENCHMARK_DATA_JSON_PATH',
    'BENCHMARK_README_DIR_ZH',
    'BENCHMARK_README_DIR_EN',
    'INDEX_DIR_ZH',
    'INDEX_DIR_EN',
    'DOCS_DIR',
    'compute_content_hash',
    'load_benchmark_data',
    'save_benchmark_data',
    'get_benchmark_entry',
    'needs_translation_update',
    'get_benchmarks_needing_translation',
]
