# Copyright (c) Alibaba, Inc. and its affiliates.
"""Load pre-computed pruned sample indices."""

import json
from pathlib import Path
from typing import Dict, List

from evalscope.utils.logger import get_logger

logger = get_logger()


def load_pruned_indices(benchmark: str) -> List[int]:
    """
    Load pre-computed pruned sample indices for a benchmark.

    Args:
        benchmark: Benchmark name ('aa_lcr' or 'live_code_bench')

    Returns:
        List of selected sample indices

    Raises:
        FileNotFoundError: If index file doesn't exist
        ValueError: If benchmark name is invalid
    """
    if benchmark not in ['aa_lcr', 'live_code_bench']:
        raise ValueError(f"Invalid benchmark: {benchmark}. Must be 'aa_lcr' or 'live_code_bench'")

    data_dir = Path(__file__).parent / 'data'
    index_file = data_dir / f'{benchmark}_pruned_indices.json'

    if not index_file.exists():
        raise FileNotFoundError(
            f"Pruned indices not found: {index_file}. "
            f"Please ensure pruned indices have been pre-computed."
        )

    with open(index_file, 'r') as f:
        data = json.load(f)

    logger.info(
        f"Loaded {len(data['selected_indices'])} pruned indices for {benchmark} "
        f"(pruned from {data['original_size']} samples, {data['reduction_pct']:.1f}% reduction)"
    )

    return data['selected_indices']


def load_pruning_metadata(benchmark: str) -> Dict:
    """
    Load metadata about the pruning process.

    Args:
        benchmark: Benchmark name

    Returns:
        Dictionary with pruning metadata (strategy, parameters, validation results)
    """
    data_dir = Path(__file__).parent / 'data'
    index_file = data_dir / f'{benchmark}_pruned_indices.json'

    with open(index_file, 'r') as f:
        return json.load(f)
