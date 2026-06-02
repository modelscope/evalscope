# Handout B — Why This Matters and How to Use It

## What Changes for the Customer Conversation

Today, evaluating a model against LiveCodeBench takes 315 inference calls. With the pruned set, it takes **189 calls** — same signal, 40% less cost and time. For AA-LCR (long-context), we go from 100 down to **50 samples**.

Concretely: a prospect asking "is Model X good enough for our coding workload?" gets an answer in **hours instead of a day**, with the same confidence in the ranking.

## How to Run This Tomorrow

```bash
# Step 1: Run the pruned evaluation
python -m evalscope_ext.tools.compare_runs \
    --review-dir ./Evals/Part\ 1/reviews \
    --benchmark live_code_bench_v5 \
    --score-key pass \
    --prune-ratio 0.5

# Step 2: For a new model, evaluate only the pruned indices
# The tool outputs which sample indices to run — feed those to your eval pipeline
```

The tool outputs a clear **pass/fail verdict**: does the new model rank above or below the known baselines? A sales engineer can read the one-line result without understanding the statistics.

## What the Multimodal Probe Gives That Random Sampling Cannot

Random sampling picks ~50 image-heavy and ~50 text-heavy MMMU questions. If a model's image encoder is degraded but its text reasoning is strong, random sampling **averages away the problem** — the model looks fine overall while silently failing on visual tasks.

Our probe **specifically targets image-encoder stress cases**: circuit diagrams, molecular structures, medical scans. If a model scores well on this probe, its encoder handles real visual complexity. If it fails, we catch it before the customer discovers it in production.

## Why a PM Should Care

1. **Faster deal cycles** — "We can validate your model in a few hours" is a competitive differentiator
2. **Confidence without cost** — The pruned set is validated to produce the same ranking as the full benchmark
3. **Multimodal readiness** — When the customer asks "can this handle images?", we have a 50-question probe that gives a direct answer, not a week-long evaluation
4. **Reproducibility** — The pruning strategy is deterministic and documented; any engineer can re-run it and get the same result
