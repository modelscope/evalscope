# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Generate dataset documentation for EvalScope benchmarks.

This module provides functions for:
1. Generating index pages with tables linking to individual benchmark READMEs
2. Generating individual README files for each benchmark from persisted data
3. Updating benchmark data (metadata, statistics, sample examples)

Note: This is a library module. CLI operations are handled by evalscope benchmark-info command.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from . import (
    BENCHMARK_DATA_JSON_PATH,
    BENCHMARK_README_DIR_EN,
    BENCHMARK_README_DIR_ZH,
    INDEX_DIR_EN,
    INDEX_DIR_ZH,
    compute_content_hash,
    get_benchmark_entry,
    load_benchmark_data,
    save_benchmark_data,
)

# Set BUILD_DOC to avoid heavy dependencies during doc generation
os.environ.setdefault('BUILD_DOC', '1')

# =============================================================================
# Localization Dictionaries
# =============================================================================


def get_index_locale(category: str, lang: str) -> Dict[str, str]:
    """Get localized strings for index page."""
    locale = {
        'title': {
            'zh': f'{category}评测集',
            'en': f'{category} Benchmarks'
        },
        'intro': {
            'zh': f'以下是支持的{category}评测集列表，点击数据集名称可查看详细信息。',
            'en': f'Below is the list of supported {category} benchmarks. Click on a benchmark name for details.'
        },
        'name': {
            'zh': '数据集名称',
            'en': 'Benchmark Name'
        },
        'pretty_name': {
            'zh': '标准名称',
            'en': 'Pretty Name'
        },
        'tags': {
            'zh': '任务类别',
            'en': 'Task Categories'
        },
        'samples': {
            'zh': '样本数',
            'en': 'Samples'
        },
    }
    return {k: v[lang] for k, v in locale.items()}


# =============================================================================
# README Generation
# =============================================================================


def generate_readme_content(
    name: str,
    meta: Dict[str, Any],
    statistics: Dict[str, Any],
    sample_example: Dict[str, Any],
    lang: str = 'en',
) -> str:
    """
    Generate complete README content for a benchmark.

    Args:
        name: Benchmark name
        meta: Benchmark metadata
        statistics: Data statistics
        sample_example: Sample example data
        lang: Language code ('en' or 'zh')

    Returns:
        Formatted README markdown string
    """
    pretty_name = meta.get('pretty_name') or name
    dataset_id = meta.get('dataset_id', '')
    description = meta.get('description', '')
    paper_url = meta.get('paper_url', '')
    tags = meta.get('tags', [])
    metrics = meta.get('metrics', [])
    few_shot_num = meta.get('few_shot_num', 0)
    eval_split = meta.get('eval_split', '')
    subset_list = meta.get('subset_list', [])
    prompt_template = meta.get('prompt_template', '')
    system_prompt = meta.get('system_prompt', '')

    # Format dataset ID link
    if dataset_id.startswith(('http://', 'https://')):
        dataset_id_md = f'[{os.path.basename(dataset_id)}]({dataset_id})'
    elif '/' in dataset_id:
        dataset_id_md = f'[{dataset_id}](https://modelscope.cn/datasets/{dataset_id}/summary)'
    else:
        dataset_id_md = f'`{dataset_id}`' if dataset_id else 'N/A'

    # Paper link
    paper_link = f'[Paper]({paper_url})' if paper_url else 'N/A'

    # Format tags and metrics
    tags_str = ', '.join(f'`{t}`' for t in tags) if tags else 'N/A'
    metrics_str = ', '.join(
        f'`{m}`' if isinstance(m, str) else f'`{list(m.keys())[0]}`' for m in metrics
    ) if metrics else 'N/A'

    # Build README content
    lines = [
        f'# {pretty_name}',
        '',
        description if description else '*No description available.*',
        '',
        '## Overview',
        '',
        '| Property | Value |',
        '|----------|-------|',
        f'| **Benchmark Name** | `{name}` |',
        f'| **Dataset ID** | {dataset_id_md} |',
        f'| **Paper** | {paper_link} |',
        f'| **Tags** | {tags_str} |',
        f'| **Metrics** | {metrics_str} |',
        f'| **Default Shots** | {few_shot_num}-shot |',
    ]

    if eval_split:
        lines.append(f'| **Evaluation Split** | `{eval_split}` |')

    # Data statistics section
    lines.extend(['', '## Data Statistics', ''])
    if statistics:
        total_samples = statistics.get('total_samples', 'N/A')
        prompt_length = statistics.get('prompt_length', {})
        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        if isinstance(total_samples, int):
            lines.append(f'| Total Samples | {total_samples:,} |')
        else:
            lines.append(f'| Total Samples | {total_samples} |')
        if prompt_length:
            lines.append(f'| Prompt Length (Mean) | {prompt_length.get("mean", "N/A")} chars |')
            lines.append(
                f'| Prompt Length (Min/Max) | {prompt_length.get("min", "N/A")} / {prompt_length.get("max", "N/A")} chars |'  # noqa: E501
            )

        # Subset statistics
        subset_stats = statistics.get('subset_stats', [])
        if subset_stats and len(subset_stats) > 1:
            lines.extend(['', '**Per-Subset Statistics:**', ''])
            lines.append('| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |')
            lines.append('|--------|---------|-------------|------------|------------|')
            for s in subset_stats:
                sample_count = s.get('sample_count', 'N/A')
                count_str = f'{sample_count:,}' if isinstance(sample_count, int) else str(sample_count)
                lines.append(
                    f'| `{s.get("name", "N/A")}` | {count_str} | '
                    f'{s.get("prompt_length_mean", "N/A")} | {s.get("prompt_length_min", "N/A")} | '
                    f'{s.get("prompt_length_max", "N/A")} |'
                )
    else:
        lines.append('*Statistics not available.*')

    # Subsets section
    lines.extend(['', '## Subsets', ''])
    if subset_list:
        subset_stats = statistics.get('subset_stats', []) if statistics else []
        subset_counts = {s.get('name'): s.get('sample_count', 'N/A') for s in subset_stats}

        if len(subset_list) == 1:
            count = subset_counts.get(subset_list[0], 'N/A')
            count_str = f'{count:,}' if isinstance(count, int) else str(count)
            lines.append(f'- `{subset_list[0]}` ({count_str} samples)')
        else:
            lines.append('| Subset | Sample Count |')
            lines.append('|--------|--------------|')
            for subset in subset_list:
                count = subset_counts.get(subset, 'N/A')
                count_str = f'{count:,}' if isinstance(count, int) else str(count)
                lines.append(f'| `{subset}` | {count_str} |')
    else:
        lines.append('*No subsets defined.*')

    # Sample example section
    lines.extend(['', '## Sample Example', ''])
    if sample_example and sample_example.get('data'):
        subset = sample_example.get('subset')
        if subset:
            lines.append(f'**Subset**: `{subset}`')
            lines.append('')
        lines.append('```json')
        lines.append(json.dumps(sample_example.get('data', {}), ensure_ascii=False, indent=2))
        lines.append('```')
        if sample_example.get('truncated'):
            lines.append('')
            lines.append('*Note: Some content was truncated for display.*')
    else:
        lines.append('*Sample example not available.*')

    # Prompt template section
    lines.extend(['', '## Prompt Template', ''])
    if system_prompt:
        lines.append('**System Prompt:**')
        lines.append('```text')
        lines.append(system_prompt)
        lines.append('```')
        lines.append('')
    if prompt_template:
        lines.append('**Prompt Template:**')
        lines.append('```text')
        lines.append(prompt_template)
        lines.append('```')
    else:
        lines.append('*No prompt template defined.*')

    # Usage section
    lines.extend([
        '', '## Usage', '', '```python', 'from evalscope import run_task', '', 'results = run_task(',
        f"    model='your-model',", f"    datasets=['{name}'],", ')', '```'
    ])

    return '\n'.join(lines)


def wrap_keywords(keywords: Union[str, List[str]]) -> str:
    """Convert keywords list to markdown formatted string."""
    if isinstance(keywords, str):
        return f'`{keywords}`'
    return ', '.join(sorted([f'`{keyword}`' for keyword in keywords]))


# =============================================================================
# Index Table Generation
# =============================================================================


def generate_index_table(
    benchmarks: List[Dict[str, Any]],
    category: str,
    lang: str = 'zh',
) -> str:
    """
    Generate an index table with links to individual benchmark README files.

    Args:
        benchmarks: List of benchmark data dicts with 'name' and entry data
        category: Category name
        lang: Language code

    Returns:
        Markdown index page content
    """
    text = get_index_locale(category, lang)

    lines = [
        f'# {text["title"]}',
        '',
        text['intro'],
        '',
        f'| {text["name"]} | {text["pretty_name"]} | {text["tags"]} | {text["samples"]} |',
        '|------------|----------|----------|----------|',
    ]

    for benchmark in benchmarks:
        name = benchmark['name']
        entry = benchmark['entry']
        meta = entry.get('meta', {})
        stats = entry.get('statistics', {})

        pretty_name = meta.get('pretty_name', name)
        tags = wrap_keywords(meta.get('tags', []))

        total_samples = stats.get('total_samples', 'N/A')
        samples_str = f'{total_samples:,}' if isinstance(total_samples, int) else str(total_samples)

        # Link to individual README file
        readme_link = f'../../benchmarks/{name}.md'
        lines.append(f'| `{name}` | [{pretty_name}]({readme_link}) | {tags} | {samples_str} |')

    return '\n'.join(lines)


# =============================================================================
# Adapter Data Extraction
# =============================================================================


def get_adapters():
    """Get all registered DataAdapters grouped by category."""
    from evalscope.api.benchmark import AgentAdapter, ImageEditAdapter, Text2ImageAdapter, VisionLanguageAdapter
    from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    print('Loading registered benchmarks...')
    adapters = defaultdict(list)

    iterator = BENCHMARK_REGISTRY.values()
    if use_tqdm:
        iterator = tqdm(iterator, desc='Loading Benchmarks')

    for benchmark in iterator:
        try:
            adapter = get_benchmark(benchmark.name)
            if isinstance(adapter, (Text2ImageAdapter, ImageEditAdapter)):
                adapters['aigc'].append(adapter)
            elif isinstance(adapter, VisionLanguageAdapter):
                adapters['vlm'].append(adapter)
            elif isinstance(adapter, AgentAdapter):
                adapters['agent'].append(adapter)
            else:
                adapters['llm'].append(adapter)
        except Exception as e:
            print(f'Warning: Failed to load {benchmark.name}: {e}')

    return adapters


def extract_adapter_meta(adapter) -> Dict[str, Any]:
    """Extract metadata from a DataAdapter instance."""
    meta = adapter._benchmark_meta
    return {
        'pretty_name': getattr(meta, 'pretty_name', None) or adapter.name,
        'dataset_id': getattr(meta, 'dataset_id', ''),
        'paper_url': getattr(meta, 'paper_url', None),
        'tags': list(getattr(meta, 'tags', [])) if getattr(meta, 'tags', None) else [],
        'metrics': list(getattr(meta, 'metric_list', [])) if getattr(meta, 'metric_list', None) else [],
        'few_shot_num': getattr(meta, 'few_shot_num', 0),
        'eval_split': getattr(meta, 'eval_split', ''),
        'subset_list': list(getattr(meta, 'subset_list', [])) if getattr(meta, 'subset_list', None) else [],
        'description': getattr(meta, 'description', '') or '',
        'prompt_template': getattr(meta, 'prompt_template', '') or '',
        'system_prompt': getattr(meta, 'system_prompt', '') or '',
    }


def compute_adapter_statistics(adapter, max_samples: int = 5000) -> Dict[str, Any]:
    """Compute statistics for a DataAdapter."""
    from evalscope.utils.doc_utils.benchmark_stats import compute_benchmark_statistics

    try:
        stats = compute_benchmark_statistics(adapter, max_samples_per_subset=max_samples)
        return stats.to_dict()
    except Exception as e:
        print(f'Warning: Failed to compute statistics for {adapter.name}: {e}')
        return {}


def get_adapter_sample_example(adapter, max_length: int = 500) -> Dict[str, Any]:
    """Get sample example from a DataAdapter."""
    from evalscope.utils.doc_utils.benchmark_stats import get_sample_example

    try:
        example = get_sample_example(adapter, max_length=max_length)
        if example:
            return {
                'data': example.data,
                'subset': example.subset,
                'truncated': example.truncated,
            }
    except Exception as e:
        print(f'Warning: Failed to get sample example for {adapter.name}: {e}')
    return {}


# =============================================================================
# Update Benchmark Data
# =============================================================================


def update_benchmark_data(
    benchmark_name: Optional[str] = None,
    force: bool = False,
    compute_stats: bool = True,
    max_samples: int = 5000,
) -> Dict[str, Any]:
    """
    Update benchmark data in the unified JSON file.

    Args:
        benchmark_name: Specific benchmark to update, or None for all
        force: Force recompute even if data exists
        compute_stats: Whether to compute statistics (requires dataset download)
        max_samples: Maximum samples per subset for statistics computation

    Returns:
        Updated benchmark data dict
    """
    from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

    data = load_benchmark_data()

    if benchmark_name:
        names = [benchmark_name]
    else:
        names = list(BENCHMARK_REGISTRY.keys())

    for name in names:
        if name not in BENCHMARK_REGISTRY:
            print(f'Warning: Benchmark {name} not found in registry')
            continue

        print(f'Updating {name}...')

        try:
            adapter = get_benchmark(name)
            entry = get_benchmark_entry(name, data)

            # Always update metadata
            entry['meta'] = extract_adapter_meta(adapter)
            print(f'  - Metadata updated')

            # Compute statistics if requested and not exists (or forced)
            if compute_stats and (force or not entry.get('statistics')):
                print(f'  - Computing statistics...')
                entry['statistics'] = compute_adapter_statistics(adapter, max_samples=max_samples)
                entry['sample_example'] = get_adapter_sample_example(adapter)
                print(f'  - Statistics computed')
            elif not compute_stats:
                print(f'  - Skipping statistics computation')
            else:
                print(f'  - Statistics already exist (use --force to recompute)')

            # Generate English README
            readme_en = generate_readme_content(
                name,
                entry['meta'],
                entry.get('statistics', {}),
                entry.get('sample_example', {}),
                lang='en',
            )

            # Check if translation needs update
            old_hash = entry.get('readme', {}).get('content_hash', '')
            new_hash = compute_content_hash(readme_en)

            entry['readme'] = entry.get('readme', {})
            entry['readme']['en'] = readme_en
            entry['readme']['content_hash'] = new_hash

            # Mark translation as needed if hash changed
            if old_hash != new_hash:
                entry['readme']['needs_translation'] = True
                print(f'  - README updated (translation needed)')
            else:
                print(f'  - README unchanged')

            entry['updated_at'] = datetime.now().isoformat()
            data[name] = entry

        except Exception as e:
            print(f'Error updating {name}: {e}')
            import traceback
            traceback.print_exc()

    save_benchmark_data(data)
    print(f'Saved benchmark data to {BENCHMARK_DATA_JSON_PATH}')

    return data


# =============================================================================
# Documentation Generation
# =============================================================================


def generate_docs(data: Optional[Dict[str, Any]] = None):
    """
    Generate all documentation from persisted benchmark data.

    Args:
        data: Pre-loaded benchmark data, or None to load from file
    """
    if data is None:
        data = load_benchmark_data()

    if not data:
        print('No benchmark data found. Run `evalscope benchmark-info --all --update` first.')
        return

    # Create output directories
    BENCHMARK_README_DIR_ZH.mkdir(parents=True, exist_ok=True)
    BENCHMARK_README_DIR_EN.mkdir(parents=True, exist_ok=True)
    INDEX_DIR_ZH.mkdir(parents=True, exist_ok=True)
    INDEX_DIR_EN.mkdir(parents=True, exist_ok=True)

    # Group benchmarks by category
    categories = defaultdict(list)

    for name, entry in data.items():
        meta = entry.get('meta', {})
        tags = meta.get('tags', [])

        # Determine category based on tags
        if any(t in ['text2image', 'image_generation', 'image_edit'] for t in tags):
            category = 'aigc'
        elif any(t in ['vlm', 'vision', 'multimodal'] for t in tags):
            category = 'vlm'
        elif any(t in ['agent', 'tool_use'] for t in tags):
            category = 'agent'
        else:
            category = 'llm'

        categories[category].append({'name': name, 'entry': entry})

    # Generate documentation for each category
    for category, benchmarks in categories.items():
        category_upper = category.upper()
        benchmarks.sort(key=lambda x: x['name'])

        print(f'Generating {category_upper} documentation ({len(benchmarks)} benchmarks)...')

        # Generate individual README files
        for benchmark in benchmarks:
            name = benchmark['name']
            entry = benchmark['entry']
            readme = entry.get('readme', {})

            # English README
            en_content = readme.get('en', '')
            if en_content:
                readme_path = BENCHMARK_README_DIR_EN / f'{name}.md'
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(en_content)

            # Chinese README (use translated version if available, else English)
            zh_content = readme.get('zh', '') or en_content
            if zh_content:
                readme_path = BENCHMARK_README_DIR_ZH / f'{name}.md'
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(zh_content)

        # Generate index pages
        index_zh = generate_index_table(benchmarks, category_upper, 'zh')
        index_en = generate_index_table(benchmarks, category_upper, 'en')

        # Write index pages
        with open(INDEX_DIR_ZH / f'{category}.md', 'w', encoding='utf-8') as f:
            f.write(index_zh)

        with open(INDEX_DIR_EN / f'{category}.md', 'w', encoding='utf-8') as f:
            f.write(index_en)

        print(f'  {category_upper}: {len(benchmarks)} benchmarks')

    print('Documentation generation complete.')


# Export main functions
__all__ = [
    'generate_readme_content',
    'generate_index_table',
    'generate_docs',
    'update_benchmark_data',
    'extract_adapter_meta',
    'compute_adapter_statistics',
    'get_adapter_sample_example',
    'get_adapters',
]
