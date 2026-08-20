# PhyX-MC


## Overview

PhyX is the first large-scale benchmark for physical reasoning in realistic, visually grounded
scenarios. This is its multiple-choice variant: each university-level physics problem is presented
with a figure and four answer options, and the model has to name the correct option letter.

## Task Description

- **Task Type**: Visual multiple-choice physics problem solving
- **Input**: A figure plus the problem description, question and four labelled options
- **Output**: A single option letter (A, B, C or D)
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

- The official prompt is reproduced verbatim, including its instruction to answer with the option
  letter only, so scores stay comparable with the published numbers.

## Evaluation Notes

- Primary metric: `acc`, mean over problems, reported overall and per domain.
- Default scoring is the official string-level match: the chosen letter is extracted from the reply
  and compared with the ground truth, accepting replies that mark the correct option the way the
  prompt prints it (`D:`) or emphasises it (`**D**`).
- Setting `judge.strategy='llm'` with `judge.models` reproduces the official LLM-judged mode.
  The judge is only consulted for replies whose option letter could not be extracted, matching
  upstream.
- Figures are sent inline as base64 and the largest is ~5 MB; set `max_image_bytes` in `dataset_args`
  if the served model enforces a smaller per-image limit.
- Resources: [Paper](https://arxiv.org/abs/2505.15929) | [GitHub](https://github.com/NastyMarcus/PhyX)
  | [Project page](https://killthefullmoon.github.io/projects/PhyX/index.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `phyx_mc` |
| **Dataset ID** | [evalscope/PhyX](https://modelscope.cn/datasets/evalscope/PhyX/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2505.15929) |
| **Tags** | `MCQ`, `MultiModal`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,000 |
| Prompt Length (Mean) | 487.19 chars |
| Prompt Length (Min/Max) | 178 / 2039 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `mechanics` | 550 | 471.63 | 203 | 1364 |
| `electromagnetism` | 550 | 466.88 | 189 | 1125 |
| `thermodynamics` | 500 | 498.81 | 178 | 1283 |
| `waves_acoustics` | 500 | 492.87 | 196 | 1880 |
| `optics` | 500 | 478.61 | 194 | 1376 |
| `modern_physics` | 400 | 525.59 | 199 | 2039 |

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
      "id": "4334f3a0",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~35.6KB]"
        },
        {
          "text": "A patient with a dislocated shoulder is put into a traction apparatus as shown in figure. The pulls $\\vec{A}$ and $\\vec{B} must combine to produce an outward traction force of 12.8 N on the patient’s arm. How large should these pulls be?Please directly answer the question and provide the correct OPTION LETTER ONLY, e.g., A, B, C, D. OPTION: A: 7.55N B: 5.55N C: 7.65N D: 6.65N"
        }
      ]
    }
  ],
  "target": "A",
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
    --datasets phyx_mc \
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
    datasets=['phyx_mc'],
    dataset_args={
        'phyx_mc': {
            # subset_list: ['mechanics', 'electromagnetism', 'thermodynamics']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
