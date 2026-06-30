"""
Item-level statistics computation from historical review data.

Computes difficulty, discrimination (variance), and point-biserial correlation
for each benchmark sample across multiple models.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ItemStats:
    """Per-item statistics computed from cross-model score data."""
    index: int
    difficulty: float  # mean score across models (0=all fail, 1=all pass)
    variance: float  # score variance across models (0=no discrimination)
    n_models: int  # number of models this item was evaluated on
    scores: List[float] = field(default_factory=list)

    @property
    def discrimination(self) -> float:
        """Normalized discrimination power (0 to 1). Peak at difficulty=0.5."""
        return self.variance / 0.25 if self.variance > 0 else 0.0

    @property
    def difficulty_bucket(self) -> str:
        if self.difficulty >= 0.8:
            return 'easy'
        elif self.difficulty <= 0.2:
            return 'hard'
        else:
            return 'medium'


def load_score_matrix_from_reviews(
    review_dir: str,
    benchmark_prefix: str,
    score_key: str = 'pass',
) -> Dict[int, ItemStats]:
    """
    Load per-sample scores from review JSONL files and compute item statistics.

    Args:
        review_dir: Path to directory containing review JSONL files.
        benchmark_prefix: Filename prefix (e.g., 'live_code_bench_v5' or 'aa_lcr').
        score_key: Key within sample_score.score.value (e.g., 'pass' or 'acc').

    Returns:
        Dict mapping sample index to ItemStats.
    """
    score_matrix: Dict[int, List[float]] = {}
    review_path = Path(review_dir)

    for fpath in sorted(review_path.glob(f'{benchmark_prefix}__*.jsonl')):
        with open(fpath, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = row['index']
                score = row['sample_score']['score']['value'][score_key]
                if idx not in score_matrix:
                    score_matrix[idx] = []
                score_matrix[idx].append(float(score))

    items: Dict[int, ItemStats] = {}
    for idx, scores in score_matrix.items():
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        items[idx] = ItemStats(
            index=idx,
            difficulty=mean,
            variance=variance,
            n_models=n,
            scores=scores,
        )

    return items


def load_score_matrix_from_jsonl_files(
    file_paths: List[str],
    score_key: str = 'pass',
) -> Dict[int, ItemStats]:
    """
    Load from explicit file paths (alternative to directory-based loading).

    Args:
        file_paths: List of paths to review JSONL files.
        score_key: Key within sample_score.score.value.

    Returns:
        Dict mapping sample index to ItemStats.
    """
    score_matrix: Dict[int, List[float]] = {}

    for fpath in file_paths:
        with open(fpath, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = row['index']
                score = row['sample_score']['score']['value'][score_key]
                if idx not in score_matrix:
                    score_matrix[idx] = []
                score_matrix[idx].append(float(score))

    items: Dict[int, ItemStats] = {}
    for idx, scores in score_matrix.items():
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        items[idx] = ItemStats(
            index=idx,
            difficulty=mean,
            variance=variance,
            n_models=n,
            scores=scores,
        )

    return items
