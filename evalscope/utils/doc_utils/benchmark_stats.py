# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Utility functions for computing benchmark statistics and extracting sample examples.

This module provides tools to analyze benchmark datasets, compute statistics
about prompt lengths, sample counts, and extract representative examples
for documentation purposes.
"""

import statistics as stats_module
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

from evalscope.api.messages import ChatMessage
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.benchmark.statistics import DataStatistics, SampleExample, SubsetStatistics
    from evalscope.api.dataset import DatasetDict, Sample

logger = get_logger()


def compute_text_length(text) -> int:
    """
    Compute the character length of text content.

    Args:
        text: String or list of ChatMessage objects.

    Returns:
        Total character count.
    """
    if isinstance(text, str):
        return len(text)
    elif isinstance(text, list):
        # For chat messages, concatenate all content
        total_length = 0
        for msg in text:
            if isinstance(msg, ChatMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if isinstance(content, str):
                    total_length += len(content)
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if isinstance(item, str):
                            total_length += len(item)
                        elif hasattr(item, 'text'):
                            total_length += len(item.text)
            else:
                total_length += len(str(msg))
        return total_length
    return len(str(text))


def compute_sample_lengths(sample: 'Sample') -> Tuple[int, int]:
    """
    Compute prompt and target lengths for a sample.

    Args:
        sample: Sample object.

    Returns:
        Tuple of (prompt_length, target_length).
    """
    prompt_length = compute_text_length(sample.input)

    # Handle target
    if isinstance(sample.target, list):
        target_length = sum(len(str(t)) for t in sample.target)
    else:
        target_length = len(str(sample.target)) if sample.target else 0

    return prompt_length, target_length


def compute_benchmark_statistics(
    adapter: 'DataAdapter',
    max_samples_per_subset: Optional[int] = None,
    compute_target_stats: bool = True,
) -> 'DataStatistics':
    """
    Compute comprehensive statistics for a benchmark dataset.

    This function loads the benchmark dataset and computes various statistics
    including sample counts, prompt lengths, and optionally target lengths.

    Args:
        adapter: The DataAdapter instance for the benchmark.
        max_samples_per_subset: Maximum samples to analyze per subset (for large datasets).
                               None means analyze all samples.
        compute_target_stats: Whether to compute target/answer length statistics.

    Returns:
        DataStatistics object with computed metrics.

    Example:
        >>> from evalscope.api.registry import get_benchmark
        >>> adapter = get_benchmark('gsm8k')
        >>> stats = compute_benchmark_statistics(adapter)
        >>> print(f"Total samples: {stats.total_samples}")
    """
    logger.info(f'Computing statistics for benchmark: {adapter.name}')

    # Ensure adapter has necessary configuration for loading
    # Create a minimal task config if not present
    if adapter._task_config is None:
        try:
            from evalscope.config import TaskConfig
            adapter._task_config = TaskConfig(model='dummy', datasets=[adapter.name])
        except Exception as e:
            logger.warning(f'Could not create TaskConfig for {adapter.name}: {e}')
            return DataStatistics(computed_at=datetime.now().isoformat())

    # We need to load raw data without post-processing to get accurate statistics
    # Use a minimal approach to avoid prompt template expansion
    try:
        test_dataset = adapter.load_dataset()
    except Exception as e:
        logger.warning(f'Failed to load dataset for {adapter.name}: {e}')
        return DataStatistics(computed_at=datetime.now().isoformat())

    all_prompt_lengths: List[int] = []
    all_target_lengths: List[int] = []
    subset_stats_list: List[SubsetStatistics] = []

    for subset_name, dataset in test_dataset.items():
        samples = list(dataset)

        # Apply sample limit if specified
        if max_samples_per_subset and len(samples) > max_samples_per_subset:
            samples_to_analyze = samples[:max_samples_per_subset]
        else:
            samples_to_analyze = samples

        prompt_lengths: List[int] = []
        target_lengths: List[int] = []

        for sample in samples_to_analyze:
            p_len, t_len = compute_sample_lengths(sample)
            prompt_lengths.append(p_len)
            if compute_target_stats:
                target_lengths.append(t_len)

        all_prompt_lengths.extend(prompt_lengths)
        all_target_lengths.extend(target_lengths)

        # Compute subset statistics
        if prompt_lengths:
            subset_stat = SubsetStatistics(
                name=subset_name,
                sample_count=len(dataset),  # Use original count, not limited
                prompt_length_mean=stats_module.mean(prompt_lengths),
                prompt_length_min=min(prompt_lengths),
                prompt_length_max=max(prompt_lengths),
                prompt_length_std=stats_module.stdev(prompt_lengths) if len(prompt_lengths) > 1 else 0.0,
                target_length_mean=stats_module.mean(target_lengths) if target_lengths else None,
            )
            subset_stats_list.append(subset_stat)

    # Compute overall statistics
    total_samples = sum(len(dataset) for dataset in test_dataset.values())

    result = DataStatistics(
        total_samples=total_samples,
        subset_stats=subset_stats_list,
        prompt_length_mean=stats_module.mean(all_prompt_lengths) if all_prompt_lengths else 0.0,
        prompt_length_min=min(all_prompt_lengths) if all_prompt_lengths else 0,
        prompt_length_max=max(all_prompt_lengths) if all_prompt_lengths else 0,
        prompt_length_std=stats_module.stdev(all_prompt_lengths) if len(all_prompt_lengths) > 1 else 0.0,
        target_length_mean=stats_module.mean(all_target_lengths) if all_target_lengths else None,
        computed_at=datetime.now().isoformat(),
    )

    logger.info(
        f'Statistics computed: {total_samples} total samples, '
        f'{len(subset_stats_list)} subsets, '
        f'mean prompt length: {result.prompt_length_mean:.1f}'
    )

    return result


def get_sample_example(
    adapter: 'DataAdapter',
    subset: Optional[str] = None,
    sample_index: int = 0,
    max_length: int = 500,
) -> Optional['SampleExample']:
    """
    Extract a representative sample example from the benchmark.

    Args:
        adapter: The DataAdapter instance for the benchmark.
        subset: Specific subset to get example from. If None, uses first available.
        sample_index: Index of sample to extract (default 0 for first sample).
        max_length: Maximum length for string values before truncation.

    Returns:
        SampleExample object with the extracted sample, or None if no samples available.

    Example:
        >>> from evalscope.api.registry import get_benchmark
        >>> adapter = get_benchmark('gsm8k')
        >>> example = get_sample_example(adapter)
        >>> print(example.to_json_block())
    """
    from evalscope.api.benchmark.statistics import SampleExample

    logger.info(f'Extracting sample example from benchmark: {adapter.name}')

    # Ensure adapter has necessary configuration for loading
    if adapter._task_config is None:
        try:
            from evalscope.config import TaskConfig
            adapter._task_config = TaskConfig(model='dummy', datasets=[adapter.name])
        except Exception as e:
            logger.warning(f'Could not create TaskConfig for {adapter.name}: {e}')
            return None

    try:
        test_dataset, _ = adapter.load()
    except Exception as e:
        logger.warning(f'Failed to load dataset for {adapter.name}: {e}')
        return None

    if not test_dataset:
        logger.warning(f'No dataset loaded for {adapter.name}')
        return None

    # Determine target subset
    if subset and subset in test_dataset.keys():
        target_subset = subset
    else:
        target_subset = next(iter(test_dataset.keys()))

    dataset = test_dataset.get(target_subset)
    if not dataset or len(dataset) == 0:
        logger.warning(f'No samples in subset {target_subset} for {adapter.name}')
        return None

    samples = list(dataset)
    if sample_index >= len(samples):
        sample_index = 0

    sample = samples[sample_index]

    return SampleExample.from_sample(
        sample,
        subset=target_subset,
        max_length=max_length,
    )


def compute_and_attach_statistics(
    adapter: 'DataAdapter',
    force_recompute: bool = False,
    max_samples_per_subset: Optional[int] = None,
) -> Tuple['DataStatistics', 'SampleExample']:
    """
    Compute statistics and extract sample example, attaching them to the adapter's metadata.

    This is a convenience function that computes both statistics and sample example,
    and updates the BenchmarkMeta object with the results.

    Args:
        adapter: The DataAdapter instance for the benchmark.
        force_recompute: If True, recompute even if statistics already exist.
        max_samples_per_subset: Maximum samples to analyze per subset.

    Returns:
        Tuple of (DataStatistics, SampleExample).
    """
    meta = adapter._benchmark_meta

    # Compute statistics if needed
    if force_recompute or meta.data_statistics is None:
        meta.data_statistics = compute_benchmark_statistics(
            adapter,
            max_samples_per_subset=max_samples_per_subset,
        )

    # Get sample example if needed
    if force_recompute or meta.sample_example is None:
        meta.sample_example = get_sample_example(adapter)

    return meta.data_statistics, meta.sample_example


def load_persisted_statistics(benchmark_name: str) -> Tuple[Optional['DataStatistics'], Optional['SampleExample']]:
    """
    Load statistics and sample example from persistent storage.

    Args:
        benchmark_name: Name of the benchmark.

    Returns:
        Tuple of (DataStatistics, SampleExample), either may be None if not found.
    """
    import json
    from pathlib import Path

    from evalscope.api.benchmark.statistics import DataStatistics, SampleExample, SubsetStatistics

    # Find statistics.json file
    stats_file = Path(__file__).parent.parent.parent / 'docs' / 'scripts' / 'statistics.json'
    if not stats_file.exists():
        return None, None

    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None, None

    if benchmark_name not in data:
        return None, None

    record = data[benchmark_name]

    # Reconstruct DataStatistics
    stats_data = record.get('statistics')
    stats = None
    if stats_data:
        subset_stats = [
            SubsetStatistics(
                name=s['name'],
                sample_count=s['sample_count'],
                prompt_length_mean=s.get('prompt_length_mean', 0.0),
                prompt_length_min=s.get('prompt_length_min', 0),
                prompt_length_max=s.get('prompt_length_max', 0),
                prompt_length_std=s.get('prompt_length_std'),
                target_length_mean=s.get('target_length_mean'),
            ) for s in stats_data.get('subset_stats', [])
        ]
        stats = DataStatistics(
            total_samples=stats_data.get('total_samples', 0),
            subset_stats=subset_stats,
            prompt_length_mean=stats_data.get('prompt_length', {}).get('mean', 0.0),
            prompt_length_min=stats_data.get('prompt_length', {}).get('min', 0),
            prompt_length_max=stats_data.get('prompt_length', {}).get('max', 0),
            prompt_length_std=stats_data.get('prompt_length', {}).get('std'),
            target_length_mean=stats_data.get('target_length_mean'),
            computed_at=stats_data.get('computed_at'),
        )

    # Reconstruct SampleExample
    example_data = record.get('sample_example')
    example = None
    if example_data:
        example = SampleExample.from_dict(example_data)

    return stats, example


def save_persisted_statistics(
    benchmark_name: str,
    stats: Optional['DataStatistics'] = None,
    example: Optional['SampleExample'] = None,
) -> None:
    """
    Save statistics and sample example to persistent storage.

    Args:
        benchmark_name: Name of the benchmark.
        stats: DataStatistics to save.
        example: SampleExample to save.
    """
    import json
    from pathlib import Path

    # Find statistics.json file
    stats_file = Path(__file__).parent.parent.parent / 'docs' / 'scripts' / 'statistics.json'

    # Load existing data
    data = {}
    if stats_file.exists():
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}

    record = data.get(benchmark_name, {})

    if stats:
        record['statistics'] = stats.to_dict()
    if example:
        record['sample_example'] = example.to_dict()

    record['updated_at'] = datetime.now().isoformat()

    data[benchmark_name] = record

    # Save back to file
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f'Saved statistics for {benchmark_name}')


def load_or_compute_statistics(
    adapter: 'DataAdapter',
    force_recompute: bool = False,
    save_after_compute: bool = True,
    max_samples_per_subset: Optional[int] = None,
) -> Tuple['DataStatistics', 'SampleExample']:
    """
    Load statistics from persistent storage, or compute if not available.

    This function first tries to load from persistent storage, and only
    computes (which requires downloading the dataset) if not found or
    force_recompute is True.

    Args:
        adapter: The DataAdapter instance for the benchmark.
        force_recompute: If True, recompute even if found in storage.
        save_after_compute: If True, save computed statistics to storage.
        max_samples_per_subset: Maximum samples to analyze per subset.

    Returns:
        Tuple of (DataStatistics, SampleExample).
    """
    meta = adapter._benchmark_meta

    # Try to load from persistent storage first
    if not force_recompute:
        stats, example = load_persisted_statistics(adapter.name)
        if stats:
            meta.data_statistics = stats
        if example:
            meta.sample_example = example
        if stats and example:
            logger.info(f'Loaded statistics for {adapter.name} from persistent storage')
            return stats, example

    # Compute if not found or force_recompute
    stats, example = compute_and_attach_statistics(
        adapter,
        force_recompute=True,
        max_samples_per_subset=max_samples_per_subset,
    )

    # Save to persistent storage
    if save_after_compute and (stats or example):
        save_persisted_statistics(adapter.name, stats, example)

    return stats, example
