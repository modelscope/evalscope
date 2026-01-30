# Copyright (c) Alibaba, Inc. and its affiliates.
"""Documentation utilities for benchmark data management.

This module provides utilities for:
- Individual benchmark metadata JSON files in evalscope/benchmarks/_meta/
- Content hash for detecting changes and translation updates
- README generation and translation support
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Path to benchmark metadata directory (individual JSON files per benchmark)
DOCS_DIR = Path(__file__).parent.parent.parent.parent / 'docs'
BENCHMARK_META_DIR = Path(__file__).parent.parent.parent / 'benchmarks' / '_meta'

# Output directories for benchmark READMEs
BENCHMARK_README_DIR_ZH = DOCS_DIR / 'zh' / 'benchmarks'
BENCHMARK_README_DIR_EN = DOCS_DIR / 'en' / 'benchmarks'
INDEX_DIR_ZH = DOCS_DIR / 'zh' / 'get_started' / 'supported_dataset'
INDEX_DIR_EN = DOCS_DIR / 'en' / 'get_started' / 'supported_dataset'


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash of content for change detection."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def load_benchmark_data(benchmark_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load benchmark metadata from individual JSON files.

    Args:
        benchmark_name: Specific benchmark to load, or None to load all

    Returns:
        Dict with benchmark data. If benchmark_name is specified, returns single
        benchmark data wrapped in dict. If None, returns all benchmarks.

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
    if benchmark_name:
        # Load single benchmark
        json_path = BENCHMARK_META_DIR / f'{benchmark_name}.json'
        if not json_path.exists():
            return {benchmark_name: _create_empty_entry()}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {benchmark_name: data}
        except json.JSONDecodeError:
            return {benchmark_name: _create_empty_entry()}
    else:
        # Load all benchmarks
        result = {}
        if not BENCHMARK_META_DIR.exists():
            return result
        for json_file in BENCHMARK_META_DIR.glob('*.json'):
            name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result[name] = json.load(f)
            except json.JSONDecodeError:
                continue
        return result


def save_benchmark_data(data: Dict[str, Any], benchmark_name: Optional[str] = None):
    """
    Save benchmark metadata to individual JSON files.

    Args:
        data: Benchmark data dict. If benchmark_name is None, expects dict of
              {benchmark_name: benchmark_data}. Otherwise, expects single benchmark data.
        benchmark_name: Name of benchmark to save. If None, saves all benchmarks in data.
    """
    BENCHMARK_META_DIR.mkdir(parents=True, exist_ok=True)

    if benchmark_name:
        # Save single benchmark
        json_path = BENCHMARK_META_DIR / f'{benchmark_name}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        # Save all benchmarks from data dict
        for name, benchmark_data in data.items():
            json_path = BENCHMARK_META_DIR / f'{name}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_data, f, ensure_ascii=False, indent=2)


def _create_empty_entry() -> Dict[str, Any]:
    """Create an empty benchmark entry structure."""
    return {
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
