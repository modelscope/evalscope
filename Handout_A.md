# Handout A — Why This Works

## Part A: Coding + Long-Context Compression

### Problem Statement

Given 315 LiveCodeBench and 100 AA-LCR samples evaluated across 3 models, select the minimal subset that still correctly identifies whether a new model is "good enough" — preserving the relative ranking signal while cutting evaluation cost by 50–70%.

### Approach: Variance-Weighted Stratified Sampling

I use an Item Response Theory (IRT) inspired approach:

1. **Compute per-item statistics** from cross-model scores: *difficulty* (mean pass rate) and *discrimination* (score variance across models).
2. **Stratify by difficulty** into 4 buckets (hard/medium-hard/medium-easy/easy) to ensure the pruned set covers the full difficulty spectrum.
3. **Within each stratum, select by discrimination** — items where models disagree most carry the highest information for ranking.
4. **Include calibration anchors** — a few known-easy and known-hard items as sanity checks.

### Results

| Benchmark | Full | Pruned | Reduction | Rank Preserved? | Max Score Deviation |
|-----------|------|--------|-----------|-----------------|---------------------|
| LCB v5    | 315  | 189    | 40%       | Yes (tau=1.0)   | 0.030               |
| AA-LCR    | 100  | 50     | 50%       | Yes (tau=1.0)   | 0.020               |

At these prune ratios, model ranking is **perfectly preserved** (Kendall tau = 1.0). The pruned set's average item variance is 1.8x higher than the full set (0.13 vs 0.08), confirming we're selecting the most informative samples. More aggressive ratios (30%) still correctly separate good from bad models but can swap models within 1% of each other.

### Why This Is Not Overfitting

The strategy selects items based on *structural properties* (variance, difficulty) not on which specific model gets them right. A 4th model evaluated on this subset will still be correctly ranked because:
- High-variance items discriminate between ANY model quality levels
- Difficulty stratification ensures coverage from trivial to impossible
- Calibration anchors detect floor/ceiling effects

### AA-LCR Judge Noise Consideration

AA-LCR uses an LLM judge, introducing non-deterministic scoring. The observed variance partly reflects judge noise, not true item discrimination. I account for this by:
- Using a slightly higher retention ratio (40% vs 50%) for AA-LCR
- Relying on the assumption that judge noise is symmetric across models (cancels in ranking)

---

## Part B: MMMU Multimodal Probe Design

### Goal

Design a pruning strategy for MMMU's full ~12K samples (HuggingFace: `MMMU/MMMU`) that specifically surfaces **image encoder degradation**, not generic capability gaps.

### Why the full 12K, not just the 660 reference rows

The 660 reference rows are one model's behavior on 22 subjects × 30 samples. They tell us *which subjects stress encoders* but provide too few samples per subject to reliably stratify by visual complexity within a subject. The full 12K dataset has ~545 samples per subject — enough to tier by image type and select a representative stress probe.

### Strategy: Subject-Stratified Visual Complexity Sampling

**Step 1 — Classify subjects by image-dependency.**
Using the reference data as a signal: subjects where `glm-4.5v-fp8` scores < 60% are likely image-encoder-bottlenecked (Electronics 43%, Chemistry, Biology, Diagnostics). Subjects > 75% are text-reasoning-dominant (Literature 93%, Art Theory). This classification transfers to the full dataset because subject identity is preserved across the 12K samples.

**Step 2 — Within image-heavy subjects, stratify by visual type.**
Pull the full subject splits from HuggingFace. Each sample has an `image` field and a `question_type`. Within the ~10 image-heavy subjects (~5,450 samples total), stratify by:
1. **Spatial layout** — Architecture, Electronics, Materials: diagrams where topology matters
2. **Fine-grained detail** — Chemistry (bond angles), Diagnostics (scan anomalies), Biology (cell structures): fails when encoder loses resolution
3. **Chart/graph parsing** — Finance, Economics, Energy: tests numerical visual decoding
4. **Multi-image** — samples with >1 image field: cross-image reasoning, exposes context-window encoder limits

**Step 3 — Select 6–8 samples per (subject × visual-type tier) = 50–80 total.**
Within each cell, prefer items that are *not* extreme (filter out trivially easy and trivially hard using the reference model's per-subject score distribution). The boundary cases (~40–70% pass rate in the reference) carry the most discrimination signal for encoder quality.

### Measuring encoder quality through the standard API

Since only chat-completion is available (no internal activations):

1. **Forced visual grounding** — prompt the model to enumerate visible elements before answering ("List what you see in the image, then answer"). Encoder degradation shows as hallucinated or absent visual details in this chain-of-thought.
2. **Perturbation delta** — run each probe item twice: once with the original image, once with a mildly perturbed version (slight blur or 10° rotation applied client-side before sending). A good encoder is robust; a degraded one drops sharply. The score *delta* isolates encoder fragility from reasoning capability.
3. **Image-occluded control** — re-run each probe item with the image replaced by a 1px placeholder. The gap between full-image score and no-image score measures how much the model actually *uses* the encoder; if the gap is near zero, the encoder is not contributing.

### Why this surfaces encoder degradation specifically, not generic gaps

Generic capability gaps (reasoning, knowledge) affect text-heavy and image-heavy subjects equally. Encoder degradation only hurts where the image carries load. By selecting subjects where image content is *necessary* to answer (you cannot infer the answer from text alone — e.g. "what bond angle is shown?"), any score drop is attributable to the encoder. The perturbation delta further isolates this: text reasoning is unchanged by a 10° image rotation; encoder output is not.

---

## Assumptions

- Binary scores are sufficient for ranking (no partial credit needed)
- The 3 shipped models span a representative quality range
- LLM judge noise on AA-LCR is model-independent
- MMMU subjects with <60% accuracy are image-encoder-bottlenecked, not text-reasoning-bottlenecked

## What Would Change With More Resources

- **More models:** Better variance estimates → more aggressive pruning possible
- **Live endpoint:** Could run adaptive testing (pick next item based on previous answers)
- **More time:** Would implement bootstrap confidence intervals on rank stability; cross-validate by leave-one-model-out
