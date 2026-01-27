# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Translate benchmark README content from English to Chinese.

This module provides:
1. Full README translation (not just description)
2. Content hash-based change detection to minimize unnecessary translations
3. Concurrent translation with thread pool

Usage via CLI:
    evalscope benchmark-info --translate           # Translate benchmarks that need it
    evalscope benchmark-info --translate --force   # Force re-translate all
    evalscope benchmark-info gsm8k --translate     # Translate specific benchmark
"""

import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from . import get_benchmarks_needing_translation, load_benchmark_data, needs_translation_update, save_benchmark_data

# Set BUILD_DOC to avoid heavy dependencies
os.environ.setdefault('BUILD_DOC', '1')


def _get_client():
    """Get OpenAI client for translation."""
    try:
        from dotenv import dotenv_values
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "Translation requires 'openai' and 'python-dotenv' packages. "
            'Install them with: pip install openai python-dotenv'
        )

    env = dotenv_values('.env')
    api_key = env.get('DASHSCOPE_API_KEY') or os.environ.get('DASHSCOPE_API_KEY')

    if not api_key:
        raise ValueError('DASHSCOPE_API_KEY not found. Set it in .env file or environment variable.')

    return OpenAI(
        api_key=api_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )


def translate_readme(readme_content: str, benchmark_name: str, client=None) -> str:
    """
    Translate English README content to Chinese.

    Args:
        readme_content: English README markdown content
        benchmark_name: Name of the benchmark (for context)
        client: Optional pre-created OpenAI client

    Returns:
        Chinese README markdown content
    """
    if client is None:
        client = _get_client()

    system_prompt = """You are a professional technical translator.
Translate the following English benchmark documentation to Chinese.

Guidelines:
1. Keep all markdown formatting unchanged (headers, tables, code blocks, links)
2. Keep technical terms, benchmark names, and code identifiers in English
3. Translate naturally and fluently, not word-by-word
4. Keep the same structure and sections
5. For code blocks, only translate comments if any
6. Keep URLs unchanged
7. Return ONLY the translated markdown content, no explanations

Example translations:
- "Overview" -> "概述"
- "Data Statistics" -> "数据统计"
- "Subsets" -> "子集"
- "Sample Example" -> "样例示例"
- "Prompt Template" -> "提示模板"
- "Usage" -> "使用方法"
- "Benchmark Name" -> "基准测试名称"
- "Dataset ID" -> "数据集ID"
- "Total Samples" -> "总样本数"
- "Prompt Length" -> "提示词长度"
"""

    try:
        completion = client.chat.completions.create(
            model='qwen3-max',
            messages=[{
                'role': 'system',
                'content': system_prompt
            }, {
                'role': 'user',
                'content': f'Translate this benchmark documentation for "{benchmark_name}":\n\n{readme_content}'
            }],
            temperature=0.3,
        )
        translated = completion.choices[0].message.content.strip()

        # Convert documentation links from English to Chinese
        translated = convert_doc_links(translated)

        return translated
    except Exception as e:
        print(f'Error translating {benchmark_name}: {e}')
        return ''


def convert_doc_links(text: str) -> str:
    """Convert documentation links from English to Chinese version."""
    return text.replace('https://evalscope.readthedocs.io/en/', 'https://evalscope.readthedocs.io/zh-cn/')


def translate_benchmarks(
    benchmark_names: Optional[List[str]] = None,
    force: bool = False,
    workers: int = 4,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Translate README content for benchmarks.

    Args:
        benchmark_names: Specific benchmarks to translate, or None for all needing translation
        force: Force re-translation even if translation exists
        workers: Number of parallel workers for translation
        dry_run: If True, don't save changes

    Returns:
        Dict with translation statistics
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    data = load_benchmark_data()

    if not data:
        return {
            'total': 0,
            'translated': 0,
            'skipped': 0,
            'error': 'No benchmark data found',
        }

    # Determine which benchmarks to translate
    if benchmark_names:
        to_translate = []
        for name in benchmark_names:
            if name not in data:
                print(f'Warning: Benchmark {name} not found in data')
                continue
            entry = data[name]
            en_content = entry.get('readme', {}).get('en', '')
            if not en_content:
                print(f'Warning: No English README for {name}')
                continue
            if force or needs_translation_update(name, en_content, data):
                to_translate.append((name, en_content))
    else:
        # Get all benchmarks needing translation
        names_needing = get_benchmarks_needing_translation(data)
        to_translate = []
        for name in names_needing:
            entry = data[name]
            en_content = entry.get('readme', {}).get('en', '')
            if en_content:
                to_translate.append((name, en_content))

        # If force, add all benchmarks with English content
        if force:
            for name, entry in data.items():
                en_content = entry.get('readme', {}).get('en', '')
                if en_content and name not in [t[0] for t in to_translate]:
                    to_translate.append((name, en_content))

    if not to_translate:
        print('No benchmarks need translation.')
        return {
            'total': len(data),
            'translated': 0,
            'skipped': len(data),
            'written': False,
        }

    print(f'Translating {len(to_translate)} benchmarks...')

    client = _get_client()
    translated_count = 0
    errors = []
    now_iso = datetime.datetime.now().replace(microsecond=0).isoformat() + 'Z'

    def translate_one(name_content):
        name, en_content = name_content
        try:
            zh_content = translate_readme(en_content, name, client)
            return name, zh_content, None
        except Exception as e:
            return name, '', str(e)

    # Use thread pool for parallel translation
    if use_tqdm:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(translate_one, item): item[0] for item in to_translate}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Translating', unit='benchmark'):
                name, zh_content, error = future.result()
                if error:
                    errors.append((name, error))
                    print(f'Error translating {name}: {error}')
                elif zh_content:
                    entry = data[name]
                    entry['readme']['zh'] = zh_content
                    entry['readme']['needs_translation'] = False
                    entry['translation_updated_at'] = now_iso
                    translated_count += 1
    else:
        for item in to_translate:
            name, zh_content, error = translate_one(item)
            if error:
                errors.append((name, error))
                print(f'Error translating {name}: {error}')
            elif zh_content:
                entry = data[name]
                entry['readme']['zh'] = zh_content
                entry['readme']['needs_translation'] = False
                entry['translation_updated_at'] = now_iso
                translated_count += 1
                print(f'Translated: {name}')

    # Save updated data
    written = False
    if translated_count > 0 and not dry_run:
        save_benchmark_data(data)
        written = True
        print(f'Saved {translated_count} translations.')

    result = {
        'total': len(to_translate),
        'translated': translated_count,
        'skipped': len(to_translate) - translated_count,
        'errors': errors,
        'workers': workers,
        'written': written,
    }

    return result


# Export main functions
__all__ = [
    'translate_readme',
    'translate_benchmarks',
    'convert_doc_links',
]
