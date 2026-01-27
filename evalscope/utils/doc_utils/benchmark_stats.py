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

from evalscope.api.benchmark.statistics import DataStatistics, SampleExample, SubsetStatistics
from evalscope.api.messages import ChatMessage
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
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
        test_dataset = adapter.load_dataset()
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


if __name__ == '__main__':
    # Example usage
    from evalscope.api.registry import get_benchmark
    adapter = get_benchmark('gsm8k')
    stats = compute_benchmark_statistics(adapter)
