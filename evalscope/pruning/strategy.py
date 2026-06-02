"""
Benchmark pruning strategies for evalscope.

Implements Variance-Weighted Stratified Sampling: an IRT-inspired approach
that selects the minimal sample set maximizing discrimination power while
maintaining difficulty coverage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .item_stats import ItemStats, load_score_matrix_from_reviews


class PruningStrategy(ABC):
    """Base class for benchmark pruning strategies."""

    @abstractmethod
    def select(
        self,
        item_stats: Dict[int, ItemStats],
        target_size: Optional[int] = None,
        prune_ratio: Optional[float] = None,
    ) -> List[int]:
        """
        Select a subset of sample indices to retain.

        Args:
            item_stats: Per-item statistics from historical evaluations.
            target_size: Exact number of samples to select (takes priority).
            prune_ratio: Fraction of original to keep (0.0–1.0).

        Returns:
            Sorted list of selected sample indices.
        """
        ...

    def _resolve_target(
        self,
        total: int,
        target_size: Optional[int],
        prune_ratio: Optional[float],
    ) -> int:
        if target_size is not None:
            return min(target_size, total)
        if prune_ratio is not None:
            return max(1, int(total * prune_ratio))
        return total


@dataclass
class StratumConfig:
    """Configuration for a difficulty stratum."""
    name: str
    min_difficulty: float
    max_difficulty: float
    allocation_weight: float  # relative weight for sample allocation


class VarianceStratifiedPruner(PruningStrategy):
    """
    Variance-Weighted Stratified Sampling.

    Strategy:
    1. Stratify items by difficulty into easy/medium/hard buckets.
    2. Allocate budget across strata (heavily weighted toward medium).
    3. Within each stratum, select items with highest variance (discrimination).
    4. Include calibration anchors from easy/hard extremes.

    This approach is model-agnostic: it identifies items that historically
    differentiate between models of varying quality, not items that a specific
    model gets right or wrong. The selected set generalizes to unseen models.
    """

    DEFAULT_STRATA = [
        StratumConfig('hard', 0.0, 0.25, 0.10),
        StratumConfig('medium-hard', 0.25, 0.50, 0.30),
        StratumConfig('medium-easy', 0.50, 0.75, 0.35),
        StratumConfig('easy', 0.75, 1.01, 0.25),
    ]

    def __init__(
        self,
        strata: Optional[List[StratumConfig]] = None,
        calibration_anchors: int = 2,
        min_samples_per_stratum: int = 1,
    ):
        """
        Args:
            strata: Difficulty strata configuration. Defaults to hard/medium/easy.
            calibration_anchors: Number of extreme-score anchors to always include.
            min_samples_per_stratum: Minimum samples from each stratum.
        """
        self.strata = strata or self.DEFAULT_STRATA
        self.calibration_anchors = calibration_anchors
        self.min_samples_per_stratum = min_samples_per_stratum

    def select(
        self,
        item_stats: Dict[int, ItemStats],
        target_size: Optional[int] = None,
        prune_ratio: Optional[float] = None,
    ) -> List[int]:
        total = len(item_stats)
        target = self._resolve_target(total, target_size, prune_ratio)

        if target >= total:
            return sorted(item_stats.keys())

        # Assign items to strata
        stratified: Dict[str, List[ItemStats]] = {s.name: [] for s in self.strata}
        for item in item_stats.values():
            for stratum in self.strata:
                if stratum.min_difficulty <= item.difficulty < stratum.max_difficulty:
                    stratified[stratum.name].append(item)
                    break

        # Sort each stratum by variance (descending) — most discriminative first
        for name in stratified:
            stratified[name].sort(key=lambda x: x.variance, reverse=True)

        selected: Set[int] = set()

        # Step 1: Include calibration anchors (highest and lowest difficulty items)
        all_items_sorted = sorted(item_stats.values(), key=lambda x: x.difficulty)
        for item in all_items_sorted[:self.calibration_anchors]:
            selected.add(item.index)
        for item in all_items_sorted[-self.calibration_anchors:]:
            selected.add(item.index)

        # Step 2: Ensure minimum per stratum
        for stratum in self.strata:
            items_in_stratum = stratified[stratum.name]
            for item in items_in_stratum[:self.min_samples_per_stratum]:
                selected.add(item.index)

        # Step 3: Allocate remaining budget by stratum weight
        remaining = target - len(selected)
        if remaining <= 0:
            return sorted(list(selected)[:target])

        total_weight = sum(s.allocation_weight for s in self.strata)
        for stratum in self.strata:
            allocation = max(0, int(remaining * stratum.allocation_weight / total_weight))
            items_in_stratum = stratified[stratum.name]

            added = 0
            for item in items_in_stratum:
                if item.index in selected:
                    continue
                selected.add(item.index)
                added += 1
                if added >= allocation:
                    break

        # Step 4: Fill any remaining slots with highest-variance items globally
        if len(selected) < target:
            all_by_variance = sorted(
                item_stats.values(), key=lambda x: x.variance, reverse=True
            )
            for item in all_by_variance:
                if item.index in selected:
                    continue
                selected.add(item.index)
                if len(selected) >= target:
                    break

        return sorted(selected)


def compute_item_stats(
    review_dir: str,
    benchmark_prefix: str,
    score_key: str = 'pass',
) -> Dict[int, ItemStats]:
    """Convenience wrapper for loading and computing item statistics."""
    return load_score_matrix_from_reviews(review_dir, benchmark_prefix, score_key)
