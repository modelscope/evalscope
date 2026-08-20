# PhyX-OE


## Overview

PhyX is the first large-scale benchmark for physical reasoning in realistic, visually grounded
scenarios. This is its open-ended variant: no options are shown, so the model has to derive the
answer of a university-level physics problem from the figure and state it.

## Task Description

- **Task Type**: Visual open-ended physics problem solving
- **Input**: A figure plus the problem description and question
- **Output**: A step-by-step derivation ending in the final answer (value with unit or a formula)
- **Domain**: University-level physics (mechanics, electromagnetism, thermodynamics, wave/acoustics,
  optics, modern physics)

## Key Features

- 3,000 university-level problems (`test`) over 6 core domains and 25 sub-domains, each domain exposed
  as its own subset; `eval_split='test_mini'` selects the official 1,000-problem testmini set.
- Every problem is grounded in a figure that carries information the text does not restate, so the
  model must combine visual cues with implicit physical laws.
- 6 reasoning types are represented (physical model grounding, multi-formula, spatial relation,
  numerical, predictive and implicit condition reasoning).
- Uses the default *Text-DeRedundancy* input style of the paper: the simplified problem description
  plus the question, with the figure attached.

- The official prompt is reproduced verbatim, including its request for step-by-step reasoning, so
  scores stay comparable with the published numbers.

## Evaluation Notes

- Primary metric: `acc`, mean over problems, reported overall and per domain.
- The final answer is read from `\boxed{...}`, else from a 'final answer:' / 'correct answer:'
  statement, else the whole reply is compared. A reply truncated before its answer therefore scores
  0 for reasons unrelated to physics ability; give the model a generous `generation_config.max_tokens`.
- Answers are free-form values with units, so an LLM judge is used by default (the official
  recommendation): set `judge.strategy='auto'` or `'llm'` and provide `judge.models`. The
  judge is only consulted when the answer does not already match as a string.
- `judge.strategy='rule'` falls back to the official string-level mode, which understates accuracy
  because equivalent spellings (`0.5 m` vs `50 cm`) do not match literally.
- Figures are sent inline as base64 and the largest is ~5 MB; set `max_image_bytes` in `dataset_args`
  if the served model enforces a smaller per-image limit.
- Resources: [Paper](https://arxiv.org/abs/2505.15929) | [GitHub](https://github.com/NastyMarcus/PhyX)
  | [Project page](https://killthefullmoon.github.io/projects/PhyX/index.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `phyx_oe` |
| **Dataset ID** | [evalscope/PhyX](https://modelscope.cn/datasets/evalscope/PhyX/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2505.15929) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,000 |
| Prompt Length (Mean) | 364.68 chars |
| Prompt Length (Min/Max) | 93 / 1874 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `mechanics` | 550 | 356.92 | 124 | 1273 |
| `electromagnetism` | 550 | 326.73 | 107 | 1032 |
| `thermodynamics` | 500 | 390.86 | 93 | 1174 |
| `waves_acoustics` | 500 | 379.95 | 101 | 1731 |
| `optics` | 500 | 361.15 | 109 | 1215 |
| `modern_physics` | 400 | 380.12 | 106 | 1874 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 3,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 215x46 - 5712x4953 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `mechanics`

```json
{
  "input": [
    {
      "id": "508a6723",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~35.6KB]"
        },
        {
          "text": "A patient with a dislocated shoulder is put into a traction apparatus as shown in figure. The pulls $\\vec{A}$ and $\\vec{B} must combine to produce an outward traction force of 12.8 N on the patient’s arm. How large should these pulls be? Please answer the question with step by step reasoning."
        }
      ]
    }
  ],
  "target": "7.55N",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": "0",
    "category": "Mechanics",
    "subfield": "Statics",
    "reasoning_type": [
      "Spatial Relation Reasoning"
    ]
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets phyx_oe \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['phyx_oe'],
    dataset_args={
        'phyx_oe': {
            # subset_list: ['mechanics', 'electromagnetism', 'thermodynamics']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
