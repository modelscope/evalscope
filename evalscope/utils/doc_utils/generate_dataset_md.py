# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Generate dataset documentation for EvalScope benchmarks.

This module provides functions for:
1. Generating index pages with tables linking to individual benchmark READMEs
2. Generating individual README files for each benchmark from persisted data
3. Updating benchmark data (metadata, statistics, sample examples)

Note: This is a library module. CLI operations are handled by evalscope benchmark-info command.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from . import (
    BENCHMARK_META_DIR,
    BENCHMARK_README_DIR_EN,
    BENCHMARK_README_DIR_ZH,
    INDEX_DIR_EN,
    INDEX_DIR_ZH,
    compute_content_hash,
    load_benchmark_data,
    save_benchmark_data,
)
from .readme_generator import _format_sample_count, _format_tags, generate_readme_from_dict

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
    }
    return {k: v[lang] for k, v in locale.items()}


# =============================================================================
# Index Table Generation
# =============================================================================


def wrap_keywords(keywords: Union[str, List[str]]) -> str:
    """Convert keywords list to markdown formatted string.

    This is an alias for _format_tags for backward compatibility.
    """
    return _format_tags(keywords)


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
        f'| {text["name"]} | {text["pretty_name"]} | {text["tags"]} |',
        '|------------|----------|----------|',
    ]

    for benchmark in benchmarks:
        name = benchmark['name']
        entry = benchmark['entry']
        meta = entry.get('meta', {})

        pretty_name = meta.get('pretty_name', name)
        tags = wrap_keywords(meta.get('tags', []))

        # Link to individual README file
        readme_link = f'../../benchmarks/{name}.md'
        lines.append(f'| `{name}` | [{pretty_name}]({readme_link}) | {tags} |')

    # Add hidden toctree to include all benchmark documents in the directory tree
    lines.extend([
        '',
        ':::{toctree}',
        ':hidden:',
        ':maxdepth: 1',
        '',
    ])

    # Add all benchmark files to toctree
    for benchmark in benchmarks:
        name = benchmark['name']
        lines.append(f'../../benchmarks/{name}.md')

    lines.append(':::')

    return '\n'.join(lines)


# =============================================================================
# Adapter Data Extraction
# =============================================================================


def get_adapter_category(adapter) -> str:
    """Determine the category of an adapter based on its type.

    Args:
        adapter: DataAdapter instance

    Returns:
        Category string: 'aigc', 'vlm', 'agent', or 'llm'
    """
    from evalscope.api.benchmark import AgentAdapter, ImageEditAdapter, Text2ImageAdapter, VisionLanguageAdapter

    if isinstance(adapter, (Text2ImageAdapter, ImageEditAdapter)):
        return 'aigc'
    elif isinstance(adapter, VisionLanguageAdapter):
        return 'vlm'
    elif isinstance(adapter, AgentAdapter):
        return 'agent'
    else:
        return 'llm'


def get_adapters():
    """Get all registered DataAdapters grouped by category."""
    from tqdm import tqdm

    from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

    print('Loading registered benchmarks...')
    adapters = defaultdict(list)

    for benchmark in tqdm(BENCHMARK_REGISTRY.values(), desc='Loading Benchmarks'):
        try:
            adapter = get_benchmark(benchmark.name)
            category = get_adapter_category(adapter)
            adapters[category].append(adapter)
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
        'train_split': getattr(meta, 'train_split', '') or '',
        'subset_list': list(getattr(meta, 'subset_list', [])) if getattr(meta, 'subset_list', None) else [],
        'description': getattr(meta, 'description', '') or '',
        'prompt_template': getattr(meta, 'prompt_template', '') or '',
        'system_prompt': getattr(meta, 'system_prompt', '') or '',
        'few_shot_prompt_template': getattr(meta, 'few_shot_prompt_template', '') or '',
        'aggregation': getattr(meta, 'aggregation', 'mean') or 'mean',
        'extra_params': dict(getattr(meta, 'extra_params', {})) if getattr(meta, 'extra_params', None) else {},
        'sandbox_config': dict(getattr(meta, 'sandbox_config', {})) if getattr(meta, 'sandbox_config', None) else {},
        'category': get_adapter_category(adapter),
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
    benchmark_name: Optional[Union[str, List[str]]] = None,
    force: bool = False,
    compute_stats: bool = True,
    max_samples: int = 50000,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Update benchmark data in individual JSON files using parallel processing.

    Args:
        benchmark_name: Specific benchmark name, list of names, or None for all
        force: Force recompute even if data exists
        compute_stats: Whether to compute statistics (requires dataset download)
        max_samples: Maximum samples per subset for statistics computation
        workers: Number of parallel workers (default: 8)

    Returns:
        Updated benchmark data dict
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    from evalscope.api.registry import BENCHMARK_REGISTRY

    # Handle different input types
    if benchmark_name is None:
        names = list(BENCHMARK_REGISTRY.keys())
    elif isinstance(benchmark_name, list):
        names = benchmark_name
    else:
        names = [benchmark_name]

    # Validate benchmark names
    valid_names = [name for name in names if name in BENCHMARK_REGISTRY]
    invalid_names = [name for name in names if name not in BENCHMARK_REGISTRY]

    if invalid_names:
        for name in invalid_names:
            print(f'Warning: Benchmark {name} not found in registry')

    if not valid_names:
        return {}

    # Process benchmarks in parallel using thread pool
    data = {}
    failed_benchmarks = []

    def update_single(name: str) -> tuple:
        """Update a single benchmark and return (name, result, error)."""
        try:
            result = _update_single_benchmark(name, force, compute_stats, max_samples)
            return (name, result, None)
        except Exception as e:
            import traceback
            error_msg = f'{e}\n{traceback.format_exc()}'
            return (name, None, error_msg)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(update_single, name): name for name in valid_names}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Updating benchmarks'):
            name, result, error = future.result()
            if error:
                print(f'Error updating {name}: {error}')
                failed_benchmarks.append(name)
            elif result:
                data[name] = result

    if failed_benchmarks:
        print(f'\nFailed to update benchmarks: {failed_benchmarks}')

    return data


def _update_single_benchmark(
    name: str,
    force: bool,
    compute_stats: bool,
    max_samples: int,
) -> Optional[Dict[str, Any]]:
    """
    Update a single benchmark.

    Args:
        name: Benchmark name
        force: Force recompute even if data exists
        compute_stats: Whether to compute statistics
        max_samples: Maximum samples per subset for statistics computation

    Returns:
        Updated benchmark entry or None if failed
    """
    from evalscope.api.registry import get_benchmark

    try:
        adapter = get_benchmark(name)
        # Load single benchmark data
        single_data = load_benchmark_data(name)
        entry = single_data[name]

        # Track if any changes were made
        has_changes = False

        # Check if metadata has changed
        new_meta = extract_adapter_meta(adapter)
        old_meta = entry.get('meta', {})
        if new_meta != old_meta:
            entry['meta'] = new_meta
            has_changes = True
            print(f'  [{name}] Metadata updated')
        else:
            print(f'  [{name}] Metadata unchanged')

        # Compute statistics if requested and not exists (or forced)
        if compute_stats and (force or not entry.get('statistics')):
            print(f'  [{name}] Computing statistics...')
            entry['statistics'] = compute_adapter_statistics(adapter, max_samples=max_samples)
            entry['sample_example'] = get_adapter_sample_example(adapter)
            has_changes = True
            print(f'  [{name}] Statistics computed')
        elif not compute_stats:
            print(f'  [{name}] Skipping statistics computation')
        else:
            print(f'  [{name}] Statistics already exist (use --force to recompute)')

        # Generate English README
        readme_en = generate_readme_from_dict(
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
            has_changes = True
            print(f'  [{name}] README updated (translation needed)')
        else:
            print(f'  [{name}] README unchanged')

        # Only update timestamp if there were actual changes
        if has_changes:
            entry['updated_at'] = datetime.now().isoformat()
            print(f'  [{name}] Updated timestamp')
        else:
            print(f'  [{name}] No changes detected')

        # Save individual benchmark file
        save_benchmark_data(entry, name)
        print(f'  [{name}] Saved to {BENCHMARK_META_DIR / name}.json')

        return entry

    except Exception as e:
        print(f'Error updating {name}: {e}')
        import traceback
        traceback.print_exc()
        raise


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

        # Determine category based on adapter type stored in metadata
        # Fall back to 'llm' if category is not found
        category = meta.get('category', 'llm')

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
    'generate_index_table',
    'generate_docs',
    'update_benchmark_data',
    'extract_adapter_meta',
    'compute_adapter_statistics',
    'get_adapter_sample_example',
    'get_adapters',
]
