"""
Unit tests for VarianceStratifiedPruner.

Run with:
    pytest evalscope/tests/test_pruning.py -v
"""

import pytest
try:
    from evalscope.pruning.item_stats import ItemStats
    from evalscope.pruning.strategy import VarianceStratifiedPruner, StratumConfig
except ModuleNotFoundError:
    from evalscope_ext.pruning.item_stats import ItemStats
    from evalscope_ext.pruning.strategy import VarianceStratifiedPruner, StratumConfig


def make_stats(index: int, difficulty: float, variance: float) -> ItemStats:
    return ItemStats(index=index, difficulty=difficulty, variance=variance, n_models=3)


def build_item_stats():
    """Build a representative set of 20 items spanning all difficulty buckets."""
    items = {}
    # 5 easy items (difficulty >= 0.75)
    for i in range(5):
        items[i] = make_stats(i, difficulty=0.9 - i * 0.02, variance=0.05 + i * 0.01)
    # 8 medium items (0.25 <= difficulty < 0.75)
    for i in range(5, 13):
        items[i] = make_stats(i, difficulty=0.7 - (i - 5) * 0.06, variance=0.15 + (i - 5) * 0.01)
    # 4 hard items (difficulty < 0.25)
    for i in range(13, 17):
        items[i] = make_stats(i, difficulty=0.20 - (i - 13) * 0.04, variance=0.08 + (i - 13) * 0.01)
    # 3 all-pass / all-fail items (zero variance)
    items[17] = make_stats(17, difficulty=1.0, variance=0.0)
    items[18] = make_stats(18, difficulty=0.0, variance=0.0)
    items[19] = make_stats(19, difficulty=0.5, variance=0.0)
    return items


class TestVarianceStratifiedPruner:

    def test_returns_correct_count_with_prune_ratio(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, prune_ratio=0.5)
        # Calibration anchors may push count slightly above floor(total * ratio)
        assert len(selected) <= len(items)
        assert len(selected) >= int(len(items) * 0.5)

    def test_returns_correct_count_with_target_size(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, target_size=7)
        assert len(selected) == 7

    def test_target_size_takes_priority_over_prune_ratio(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, target_size=5, prune_ratio=0.5)
        assert len(selected) == 5

    def test_selected_indices_are_valid(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, prune_ratio=0.5)
        for idx in selected:
            assert idx in items

    def test_selected_indices_are_sorted(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, prune_ratio=0.5)
        assert selected == sorted(selected)

    def test_no_duplicates(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, prune_ratio=0.8)
        assert len(selected) == len(set(selected))

    def test_returns_all_when_ratio_is_1(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, prune_ratio=1.0)
        assert len(selected) == len(items)

    def test_returns_all_when_target_exceeds_total(self):
        items = build_item_stats()
        pruner = VarianceStratifiedPruner()
        selected = pruner.select(items, target_size=999)
        assert len(selected) == len(items)

    def test_high_variance_items_preferred_within_stratum(self):
        """Items with higher variance should be selected before low-variance items
        in the same difficulty stratum."""
        items = {
            0: make_stats(0, difficulty=0.5, variance=0.25),  # max variance
            1: make_stats(1, difficulty=0.5, variance=0.10),
            2: make_stats(2, difficulty=0.5, variance=0.05),
            3: make_stats(3, difficulty=0.5, variance=0.01),  # low variance
        }
        pruner = VarianceStratifiedPruner(calibration_anchors=0, min_samples_per_stratum=0)
        selected = pruner.select(items, target_size=2)
        # index 0 (highest variance) must be selected
        assert 0 in selected

    def test_calibration_anchors_always_included(self):
        """The hardest and easiest items should always be in the selected set."""
        items = {
            0: make_stats(0, difficulty=0.01, variance=0.0),  # hardest
            1: make_stats(1, difficulty=0.50, variance=0.25),
            2: make_stats(2, difficulty=0.50, variance=0.20),
            3: make_stats(3, difficulty=0.99, variance=0.0),  # easiest
            4: make_stats(4, difficulty=0.50, variance=0.15),
        }
        pruner = VarianceStratifiedPruner(calibration_anchors=1)
        selected = pruner.select(items, target_size=3)
        assert 0 in selected  # hardest anchor
        assert 3 in selected  # easiest anchor

    def test_rank_preservation_on_synthetic_data(self):
        """Pruned subset should preserve model ranking when items are informative."""
        # Simulate 10 items with 3 models; model A > B > C quality
        # High variance items: model A passes, B and C fail (discrimination)
        items = {}
        model_scores = {
            'A': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'B': [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
            'C': [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        }
        for idx in range(10):
            scores = [model_scores[m][idx] for m in ['A', 'B', 'C']]
            mean = sum(scores) / 3
            variance = sum((s - mean) ** 2 for s in scores) / 3
            items[idx] = ItemStats(index=idx, difficulty=mean, variance=variance, n_models=3, scores=scores)

        pruner = VarianceStratifiedPruner(calibration_anchors=1)
        selected = set(pruner.select(items, target_size=5))

        # Compute scores on pruned set
        pruned_scores = {}
        for model, scores in model_scores.items():
            pruned = [scores[i] for i in selected]
            pruned_scores[model] = sum(pruned) / len(pruned) if pruned else 0

        full_scores = {m: sum(s) / len(s) for m, s in model_scores.items()}
        full_rank = sorted(full_scores, key=lambda m: full_scores[m], reverse=True)
        pruned_rank = sorted(pruned_scores, key=lambda m: pruned_scores[m], reverse=True)

        assert full_rank[0] == pruned_rank[0], 'Top model must be preserved'
