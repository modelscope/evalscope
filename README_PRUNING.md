# Task 2 — Benchmark Compression

**EvalScope commit SHA:** `de7b0b3f08c617f48a00ef09f7169dc74212a6d9`

## Overview

Variance-Weighted Stratified Sampling for benchmark pruning — an IRT-inspired approach that selects the minimal sample set maximizing discrimination power while maintaining difficulty coverage.

**Results:**
- LiveCodeBench v5: 315 → 157 samples (50% reduction), rank preserved
- AA-LCR: 100 → 40 samples (60% reduction), rank preserved

## Installation

```bash
cd task2
pip install -e .
```

Or add the `task2/` directory to your Python path.

## Usage

### Compare full vs pruned runs

```bash
python -m evalscope_ext.tools.compare_runs \
    --review-dir "./Evals/Part 1/reviews" \
    --benchmark live_code_bench_v5 \
    --score-key pass \
    --prune-ratio 0.5
```

```bash
python -m evalscope_ext.tools.compare_runs \
    --review-dir "./Evals/Part 1/reviews" \
    --benchmark aa_lcr \
    --score-key acc \
    --prune-ratio 0.4
```

### Programmatic usage

```python
from evalscope_ext.pruning import VarianceStratifiedPruner, compute_item_stats

# Compute item statistics from historical reviews
item_stats = compute_item_stats(
    review_dir='./Evals/Part 1/reviews',
    benchmark_prefix='live_code_bench_v5',
    score_key='pass'
)

# Run pruning
pruner = VarianceStratifiedPruner()
selected_indices = pruner.select(item_stats, prune_ratio=0.5)

print(f'Selected {len(selected_indices)} / {len(item_stats)} samples')
```

### Integration with evalscope

```bash
evalscope eval --model <model> --datasets live_code_bench \
    --dataset-args '{
        "live_code_bench": {
            "pruning_strategy": "variance_stratified",
            "prune_ratio": 0.5,
            "review_dir": "./Evals/Part 1/reviews"
        }
    }'
```

## Project Structure

```
task2/
├── evalscope_ext/
│   ├── __init__.py
│   ├── pruning/
│   │   ├── __init__.py
│   │   ├── item_stats.py     # Per-item statistics computation
│   │   ├── strategy.py       # Pruning strategy implementations
│   │   └── adapter.py        # EvalScope adapter integration
│   └── tools/
│       ├── __init__.py
│       └── compare_runs.py   # Full vs pruned comparison tool
├── Handout_A.md              # Technical write-up
├── Handout_B.md              # Non-technical write-up
├── setup.py
└── README.md
```

## Approach

See [Handout A](./Handout_A.md) for the full technical explanation.

**TL;DR:** We compute per-item difficulty (mean pass rate) and discrimination (score variance) from historical model evaluations. Items are stratified by difficulty, then selected by discrimination within each stratum. This selects the samples that are most informative for ranking models — regardless of which specific models are being evaluated.
