"""
Compare full and pruned evaluation runs.

Computes correlation, rank agreement, and score deviation between a full
benchmark run and a pruned subset run, validating that the pruned set
preserves the signal quality.

Usage:
    python -m evalscope.cli.pruning_compare \
        --review-dir "./Evals/Part 1/reviews" \
        --benchmark live_code_bench_v5 \
        --score-key pass \
        --prune-ratio 0.6
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from evalscope.pruning import (
    VarianceStratifiedPruner,
    compute_item_stats,
)


def compute_model_scores(
    review_dir: str,
    benchmark_prefix: str,
    score_key: str,
    indices: Optional[set] = None,
) -> Dict[str, float]:
    model_scores: Dict[str, List[float]] = {}
    review_path = Path(review_dir)

    for fpath in sorted(review_path.glob(f'{benchmark_prefix}__*.jsonl')):
        model_name = fpath.stem.replace(f'{benchmark_prefix}__', '')
        model_scores[model_name] = []

        with open(fpath, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = row['index']
                if indices is not None and idx not in indices:
                    continue
                score = row['sample_score']['score']['value'][score_key]
                model_scores[model_name].append(float(score))

    return {m: sum(s) / len(s) if s else 0.0 for m, s in model_scores.items()}


def rank_models(scores: Dict[str, float]) -> Dict[str, int]:
    sorted_models = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
    return {m: rank + 1 for rank, m in enumerate(sorted_models)}


def kendall_tau(ranks_a: Dict[str, int], ranks_b: Dict[str, int]) -> float:
    models = sorted(ranks_a.keys())
    n = len(models)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = models[i], models[j]
            diff_a = ranks_a[mi] - ranks_a[mj]
            diff_b = ranks_b[mi] - ranks_b[mj]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(
        description='Compare full vs pruned benchmark evaluation.'
    )
    parser.add_argument('--review-dir', type=str, required=True,
                        help='Path to review JSONL directory.')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='Benchmark prefix (e.g., live_code_bench_v5).')
    parser.add_argument('--score-key', type=str, default='pass',
                        help='Score key in review data (default: pass).')
    parser.add_argument('--prune-ratio', type=float, default=0.6,
                        help='Fraction of samples to keep (default: 0.6).')
    parser.add_argument('--target-size', type=int, default=None,
                        help='Exact number of samples to keep.')
    args = parser.parse_args()

    print(f'Benchmark:   {args.benchmark}')
    print(f'Score key:   {args.score_key}')
    print(f'Prune ratio: {args.prune_ratio}')
    print()

    full_scores = compute_model_scores(args.review_dir, args.benchmark, args.score_key)
    print('=== Full Benchmark Scores ===')
    for model, score in sorted(full_scores.items()):
        print(f'  {model}: {score:.4f}')
    full_ranks = rank_models(full_scores)
    print(f'  Ranking: {" > ".join(sorted(full_ranks, key=lambda m: full_ranks[m]))}')
    print()

    item_stats = compute_item_stats(args.review_dir, args.benchmark, args.score_key)
    pruner = VarianceStratifiedPruner()
    selected = pruner.select(item_stats, target_size=args.target_size, prune_ratio=args.prune_ratio)
    selected_set = set(selected)

    print('=== Pruning Results ===')
    print(f'  Total: {len(item_stats)}  Selected: {len(selected)} ({len(selected)/len(item_stats)*100:.1f}%)')
    easy = sum(1 for i in selected if item_stats[i].difficulty >= 0.8)
    medium = sum(1 for i in selected if 0.2 < item_stats[i].difficulty < 0.8)
    hard = sum(1 for i in selected if item_stats[i].difficulty <= 0.2)
    print(f'  Distribution: easy={easy}, medium={medium}, hard={hard}')
    avg_var = sum(item_stats[i].variance for i in selected) / len(selected)
    all_avg_var = sum(s.variance for s in item_stats.values()) / len(item_stats)
    print(f'  Avg variance: selected={avg_var:.4f} vs all={all_avg_var:.4f}')
    print()

    pruned_scores = compute_model_scores(args.review_dir, args.benchmark, args.score_key, selected_set)
    print('=== Pruned Subset Scores ===')
    for model, score in sorted(pruned_scores.items()):
        delta = score - full_scores[model]
        print(f'  {model}: {score:.4f} (delta={delta:+.4f})')
    pruned_ranks = rank_models(pruned_scores)
    print(f'  Ranking: {" > ".join(sorted(pruned_ranks, key=lambda m: pruned_ranks[m]))}')
    print()

    tau = kendall_tau(full_ranks, pruned_ranks)
    rank_preserved = full_ranks == pruned_ranks
    max_delta = max(abs(pruned_scores[m] - full_scores[m]) for m in full_scores)

    print('=== Comparison ===')
    print(f'  Rank preserved:    {"YES" if rank_preserved else "NO"}')
    print(f'  Kendall tau:       {tau:.4f}')
    print(f'  Max score delta:   {max_delta:.4f}')
    print(f'  Compression:       {len(item_stats)} -> {len(selected)} ({(1 - len(selected)/len(item_stats))*100:.0f}% reduction)')

    if rank_preserved:
        print(f'\n  RESULT: Pruned set preserves model ranking with {(1-len(selected)/len(item_stats))*100:.0f}% fewer samples.')
    else:
        print('\n  WARNING: Rank order changed. Consider increasing prune_ratio.')


if __name__ == '__main__':
    main()
