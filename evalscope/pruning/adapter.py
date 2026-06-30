"""
EvalScope benchmark adapter with pruning support.

This module provides a mixin and adapter classes that integrate the pruning
strategy into evalscope's dataset loading pipeline. The pruner intercepts
samples after loading and filters them to the selected subset.

Usage via evalscope CLI:
    evalscope eval --model <model> --datasets live_code_bench_pruned \
        --dataset-args '{"live_code_bench_pruned": {
            "pruning_strategy": "variance_stratified",
            "prune_ratio": 0.3,
            "review_dir": "./Evals/Part 1/reviews"
        }}'
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .item_stats import ItemStats, load_score_matrix_from_reviews
from .strategy import PruningStrategy, VarianceStratifiedPruner

logger = logging.getLogger(__name__)

# Strategy registry
STRATEGY_REGISTRY: Dict[str, type] = {
    'variance_stratified': VarianceStratifiedPruner,
}


class PruningMixin:
    """
    Mixin that adds sample pruning capability to any evalscope DataAdapter.

    Reads pruning configuration from benchmark_meta.extra_params and filters
    samples during the load phase.
    """

    def _get_pruning_config(self) -> Dict[str, Any]:
        """Extract pruning config from adapter's benchmark metadata."""
        meta = getattr(self, 'benchmark_meta', None)
        if meta is None:
            return {}

        # Config comes through dataset_args -> extra_params
        extra = getattr(meta, 'extra_params', {})
        if isinstance(extra, dict):
            return extra
        return {}

    def _build_pruner(self, config: Dict[str, Any]) -> Optional[PruningStrategy]:
        """Instantiate the pruning strategy from config."""
        strategy_name = config.get('pruning_strategy', 'none')
        if strategy_name == 'none' or not strategy_name:
            return None

        strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown pruning strategy: '{strategy_name}'. "
                f"Available: {list(STRATEGY_REGISTRY.keys())}"
            )

        return strategy_cls()

    def _compute_selected_indices(self, config: Dict[str, Any]) -> Optional[Set[int]]:
        """Compute the set of indices to retain after pruning."""
        pruner = self._build_pruner(config)
        if pruner is None:
            return None

        review_dir = config.get('review_dir')
        benchmark_prefix = config.get('benchmark_prefix', '')
        score_key = config.get('score_key', 'pass')
        prune_ratio = config.get('prune_ratio', 0.3)
        target_size = config.get('target_size')

        if not review_dir:
            logger.warning(
                "Pruning requested but 'review_dir' not provided. "
                "Skipping pruning — all samples will be used."
            )
            return None

        review_path = Path(review_dir)
        if not review_path.exists():
            logger.warning(f"Review directory not found: {review_dir}. Skipping pruning.")
            return None

        # Load item statistics from historical reviews
        item_stats = load_score_matrix_from_reviews(
            str(review_path), benchmark_prefix, score_key
        )

        if not item_stats:
            logger.warning("No item statistics computed. Skipping pruning.")
            return None

        # Run pruning strategy
        selected = pruner.select(
            item_stats,
            target_size=target_size,
            prune_ratio=prune_ratio,
        )

        logger.info(
            f"Pruning: selected {len(selected)}/{len(item_stats)} samples "
            f"(ratio={len(selected)/len(item_stats):.2f}, "
            f"strategy={config.get('pruning_strategy')})"
        )

        return set(selected)

    def filter_samples_by_pruning(self, samples: List[Dict], config: Dict[str, Any]) -> List[Dict]:
        """
        Filter a list of sample dicts to only those in the pruned set.

        Args:
            samples: List of sample dicts with 'index' field.
            config: Pruning configuration dict.

        Returns:
            Filtered list of samples.
        """
        selected = self._compute_selected_indices(config)
        if selected is None:
            return samples

        filtered = [s for s in samples if s.get('index') in selected]
        logger.info(f"Filtered {len(samples)} -> {len(filtered)} samples")
        return filtered


def get_pruned_indices(
    review_dir: str,
    benchmark_prefix: str,
    score_key: str = 'pass',
    strategy: str = 'variance_stratified',
    prune_ratio: float = 0.3,
    target_size: Optional[int] = None,
) -> List[int]:
    """
    Standalone function to compute pruned sample indices.

    Useful for scripts that need to get the pruned set without going
    through the full evalscope pipeline.

    Args:
        review_dir: Path to review JSONL files.
        benchmark_prefix: Filename prefix (e.g., 'live_code_bench_v5').
        score_key: Score field name ('pass' for LCB, 'acc' for AA-LCR).
        strategy: Pruning strategy name.
        prune_ratio: Fraction of samples to retain.
        target_size: Exact number of samples (overrides prune_ratio).

    Returns:
        Sorted list of selected sample indices.
    """
    item_stats = load_score_matrix_from_reviews(review_dir, benchmark_prefix, score_key)

    strategy_cls = STRATEGY_REGISTRY.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    pruner = strategy_cls()
    return pruner.select(item_stats, target_size=target_size, prune_ratio=prune_ratio)
