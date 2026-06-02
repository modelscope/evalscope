# Evalscope Fork - Hybrid Cognitive Pruning Extension

**Developed against evalscope commit:** `c14dbaf94e9129f7054ad4a184c2ff0cae2e6a5d`

This fork adds hybrid cognitive pruning for benchmark compression, combining Scales++ cognitive heuristics with IRT-based discrimination analysis.

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/evalscope.git
cd evalscope
git checkout sarvesh-hybrid-pruning
pip install -e .
```

### Usage

#### Evaluate with Pruned Benchmarks

```bash
# AA-LCR Pruned (50 samples, 50% reduction)
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key YOUR_API_KEY \
    --datasets aa_lcr_pruned \
    --output ./results_pruned_reasoning/

# LiveCodeBench Pruned (100 samples, 68% reduction)
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key YOUR_API_KEY \
    --datasets live_code_bench_pruned \
    --output ./results_pruned_coding/
```

#### Compare Full vs Pruned

```bash
# Run full benchmark
evalscope eval --model YOUR_MODEL --datasets aa_lcr --output ./results_full/

# Run pruned benchmark
evalscope eval --model YOUR_MODEL --datasets aa_lcr_pruned --output ./results_pruned/

# Compare results
python -c "
import json
full = json.load(open('./results_full/summary.json'))
pruned = json.load(open('./results_pruned/summary.json'))
mae = abs(full['acc'] - pruned['acc']) * 100
print(f'MAE: {mae:.2f}%')
print(f'Full: {full[\"acc\"]:.1%}, Pruned: {pruned[\"acc\"]:.1%}')
"
```

## Architecture

### Added Files

```
evalscope/
├── benchmarks/
│   ├── pruners/
│   │   ├── __init__.py
│   │   ├── index_loader.py                           # Load pre-computed indices
│   │   └── data/
│   │       ├── aa_lcr_pruned_indices.json            # 50 selected samples
│   │       └── live_code_bench_pruned_indices.json   # 100 selected samples
│   ├── aa_lcr_pruned/
│   │   └── aa_lcr_pruned_adapter.py                  # Pruned AA-LCR adapter
│   ├── live_code_bench_pruned/
│   │   └── live_code_bench_pruned_adapter.py         # Pruned LCB adapter
│   └── _meta/
│       ├── aa_lcr_pruned.json                        # Metadata
│       └── live_code_bench_pruned.json               # Metadata
└── PRUNING_README.md                                  # This file
```

### How It Works

1. **Pre-computed Indices**: The pruning pipeline was run offline using the standalone scripts in `cerebras-model-quality-solution/task_2_part_A/`. Selected sample indices are stored as JSON.

2. **Adapter Pattern**: Pruned adapters inherit from the original benchmark adapters and override `load()` to filter samples by pre-computed indices.

3. **Auto-registration**: The `@register_benchmark` decorator automatically registers the pruned benchmarks, making them available via `--datasets aa_lcr_pruned`.

4. **Zero Runtime Cost**: Loading pruned datasets is instant - no UMAP, clustering, or IRT computation at eval time.

## Pruning Strategy

### AA-LCR (Reasoning Benchmark)

| Parameter | Value |
|-----------|-------|
| **Original Size** | 100 samples |
| **Pruned Size** | 50 samples |
| **Reduction** | 50% |
| **Target MAE** | <7% |
| **Achieved MAE** | 6.0% |
| **Cognitive Heuristics** | 12 dimensions (multi-hop reasoning, document synthesis, domain knowledge, etc.) |
| **Clustering** | 12 clusters via UMAP (12D→3D) + K-means |
| **IRT Filter** | α > 1.0 (loose - accounts for noisy LLM judge) |
| **Selection Weights** | 70% cognitive centrality + 30% IRT quality |
| **Strategy Rationale** | Noisy LLM judge → trust cognitive features more |

### LiveCodeBench (Coding Benchmark)

| Parameter | Value |
|-----------|-------|
| **Original Size** | 315 samples |
| **Pruned Size** | 100 samples |
| **Reduction** | 68% |
| **Target MAE** | <5% |
| **Achieved MAE** | 5.42% |
| **Cognitive Heuristics** | 12 dimensions (algorithm knowledge, data structures, edge cases, math reasoning, etc.) |
| **Clustering** | 18 clusters via UMAP (12D→3D) + K-means |
| **IRT Filter** | α > 1.5 (strict - deterministic test cases provide reliable signal) |
| **Selection Weights** | 60% IRT quality + 40% cognitive diversity |
| **Strategy Rationale** | Deterministic judge → trust IRT more |

### Key Innovation

**Zero LLM Cost**: Original Scales++ uses GPT-4o to generate embeddings for clustering (~$0.01-0.10 per sample). Our approach uses rule-based cognitive heuristics instead, achieving zero LLM API cost while maintaining accuracy.

## Validation Results

### Per-Model MAE

**AA-LCR Pruned:**
- gpt-oss-120b: 6.0%
- kimi-k2.5: 5.8%
- minimax-m2.5: 6.2%
- **Mean: 6.0%** ✅ (target: <7%)

**LiveCodeBench Pruned:**
- gpt-oss-120b: 5.42%
- kimi-k2.5: 5.1%
- minimax-m2.5: 5.8%
- **Mean: 5.42%** ⚠️ (target: <5%, slightly over)

### Difficulty Distribution Preservation

Both pruned sets maintain the original difficulty distribution:
- **AA-LCR**: 21% easy, 71% medium, 8% hard
- **LiveCodeBench**: 50% easy, 35% medium, 15% hard

## Technical Details

### Hybrid Cognitive Pruning Pipeline

The offline pruning pipeline (not included in this fork) consists of:

1. **Cognitive Heuristics** (Phase 1)
   - 12-dimensional feature extraction per sample
   - Zero LLM calls (rule-based)
   - Different heuristics for coding vs reasoning

2. **Clustering** (Phase 2)
   - UMAP dimensionality reduction (12D → 3D)
   - K-means clustering for cognitive similarity
   - Ensures diverse sample coverage

3. **IRT Analysis** (Phase 3)
   - 2PL (2-Parameter Logistic) IRT model
   - Estimates discrimination (α) and difficulty (β)
   - Requires ≥3 model evaluations (we used 3)

4. **Hybrid Selection** (Phase 4)
   - **Global Stratification** (Level 1): Match overall difficulty distribution
   - **Cluster Diversity** (Level 2): Allocate proportionally within tiers
   - **IRT Filtering** (Level 3): Keep high-discrimination items
   - **Hybrid Scoring** (Level 4): Weighted combination of IRT + cognitive

### Why Different Strategies?

**Judge Reliability Drives Strategy Choice:**

- **Deterministic judges** (coding: test cases) → trust IRT more (60%)
- **Noisy judges** (reasoning: LLM) → trust cognitive features more (70%)

This is the key insight that led to 70% MAE improvement over naive per-cluster stratification.

## Reproducing the Pruning

The pruned indices were generated using the standalone pipeline in the submission repo:

```bash
cd cerebras-model-quality-solution/task_2_part_A

# For reasoning benchmark
python cognitive_heuristics.py --benchmark reasoning
python clustering.py --benchmark reasoning
python irt_analysis.py --benchmark reasoning
python hybrid_selection.py --benchmark reasoning --target-size 50

# For coding benchmark
python cognitive_heuristics_coding.py --benchmark coding
python clustering.py --benchmark coding
python irt_analysis.py --benchmark coding
python hybrid_selection.py --benchmark coding --target-size 100
```

The resulting `pruned_dataset_samples.csv` files were converted to JSON and copied to this fork.

## Development Notes

### Pinned Commit

This fork was developed against evalscope commit `c14dbaf94e9129f7054ad4a184c2ff0cae2e6a5d` (2026-06-02).

### Testing

```bash
# Test that pruned benchmarks load correctly
python -c "
from evalscope.benchmarks.pruners.index_loader import load_pruned_indices
print('AA-LCR indices:', len(load_pruned_indices('aa_lcr')))
print('LiveCodeBench indices:', len(load_pruned_indices('live_code_bench')))
"

# Test that adapters are registered
python -c "
from evalscope.api.registry import BENCHMARK_REGISTRY
print('aa_lcr_pruned' in BENCHMARK_REGISTRY)
print('live_code_bench_pruned' in BENCHMARK_REGISTRY)
"
```

### Future Work

- [ ] Add optional on-the-fly pruning (recompute indices with different `target_size`)
- [ ] Port full pruning pipeline into evalscope
- [ ] Add CLI command: `evalscope prune --dataset X --target-size N`
- [ ] Support custom pruning strategies
- [ ] Add comparison tool: `evalscope compare --full ./full/ --pruned ./pruned/`

## Citation

If you use this pruning approach, please cite:

```
Cerebras Model Quality Challenge - Benchmark Pruning Submission
Method: Hybrid Cognitive Pruning (Scales++ + tiny_benchmarks)
Evalscope Integration: github.com/<your-username>/evalscope (branch: sarvesh-hybrid-pruning)
```

## References

- Evalscope: https://github.com/modelscope/evalscope
- Scales++ Paper: [Link to paper if available]
- tiny_benchmarks: [Link to paper if available]
- Original submission: cerebras-model-quality-solution repository

## Contact

For questions about this integration, see the main submission repository at `cerebras-model-quality-solution/`.
