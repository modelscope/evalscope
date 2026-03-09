# Copyright (c) Alibaba, Inc. and its affiliates.
"""Documentation utilities for benchmark data management.

This module provides utilities for:
- Individual benchmark metadata JSON files in evalscope/benchmarks/_meta/
- Content hash for detecting changes and translation updates
- README generation and translation support
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from evalscope.utils.resource_utils import BENCHMARK_META_DIR
from evalscope.utils.resource_utils import _create_empty_benchmark_entry as _create_empty_entry
from evalscope.utils.resource_utils import load_benchmark_data, save_benchmark_data

# Path to docs directory
DOCS_DIR = Path(__file__).parent.parent.parent.parent / 'docs'

# Output directories for benchmark READMEs
BENCHMARK_README_DIR_ZH = DOCS_DIR / 'zh' / 'benchmarks'
BENCHMARK_README_DIR_EN = DOCS_DIR / 'en' / 'benchmarks'
INDEX_DIR_ZH = DOCS_DIR / 'zh' / 'get_started' / 'supported_dataset'
INDEX_DIR_EN = DOCS_DIR / 'en' / 'get_started' / 'supported_dataset'


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash of content for change detection."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


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
