# HiPhO


## Overview

HiPhO is the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. It compiles
13 recent Olympiad exams (2024-2025) spanning international and regional competitions, with mixed modalities that
range from text-only problems to diagram-based problems.

## Task Description

- **Task Type**: Free-form physics problem solving graded against official marking schemes
- **Input**: A physics problem (constants sheet + context + question), optionally with figures
- **Output**: A step-by-step solution ending with boxed final answers inside `<answer>...</answer>`
- **Modalities**: Text-only and text+figure (illustration / variable / data figures)

## Key Features

- 403 problems across 14 exam papers (IPhO, APhO, EuPhO, NBPhO, PanPhO, PanMechanics, CPhO, F=MA), each exam
  exposed as its own subset.
- English prompts are used for English exams and Chinese prompts for the Chinese exams (CPhO, PanMechanics),
  following the official language mapping.
- Two grading regimes reproduced from the paper, dispatched per problem:
  - **Step-level** for problems shipping an official marking scheme: the LLM judge scores every marking criterion
    and the awarded points are summed.
  - **Answer-level** for problems without a marking scheme: boxed final answers are matched against the ground
    truth by a rule-based math check, with an LLM judge as fallback.

## Evaluation Notes

- Requires an LLM judge: run with `judge_strategy='llm'` (or `'auto'`, which enables the judge for this benchmark)
  and provide `judge_model_args`. `judge_strategy='rule'` is not supported.
- Primary metric: `accuracy`, the per-problem awarded/attainable point ratio in `[0, 1]`, aggregated by mean per subset.
  For step-level problems the attainable maximum is the sum of the marking criteria; for problems with several
  official schemes (EuPhO, NBPhO) the highest-scoring scheme is used, matching the paper.
- This reports the normalized exam score per exam. It does not compute the paper's gold/silver/bronze medal
  thresholds, which require the raw point totals and official cutoffs.
- Solutions can be long and figure problems need vision input; give the evaluated model a generous
  `generation_config.max_tokens`. A solution truncated before its `<answer>` block yields no boxed answer and
  scores near zero for reasons unrelated to physics ability.
- Figures are sent inline as base64 and the largest is ~1.5 MB; set `max_image_bytes` in `dataset_args` if the
  served model enforces a smaller per-image limit.
- Resources: [Paper](https://arxiv.org/abs/2509.07894) | [GitHub](https://github.com/SciYu/HiPhO) |
  [Leaderboard](https://phyarena.github.io/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hipho` |
| **Dataset ID** | [evalscope/HiPhO](https://modelscope.cn/datasets/evalscope/HiPhO/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2509.07894) |
| **Tags** | `Math`, `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 403 |
| Prompt Length (Mean) | 3020.35 chars |
| Prompt Length (Min/Max) | 653 / 9336 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `APhO_2025` | 45 | 4624.02 | 2496 | 8787 |
| `CPhO_2025` | 43 | 2041.81 | 960 | 3745 |
| `EuPhO_2024` | 7 | 1924.29 | 1468 | 2051 |
| `EuPhO_2025` | 6 | 1646.33 | 1422 | 1856 |
| `F=MA_2024` | 25 | 1598.76 | 1279 | 1957 |
| `F=MA_2025` | 25 | 1721.2 | 1395 | 2513 |
| `IPhO_2024` | 37 | 4152.57 | 2201 | 6701 |
| `IPhO_2025` | 39 | 6359.74 | 3362 | 9336 |
| `NBPhO_2024` | 24 | 2305.25 | 1317 | 4486 |
| `NBPhO_2025` | 20 | 2677.7 | 1359 | 4808 |
| `PanMechanics_2024` | 29 | 878.55 | 653 | 1283 |
| `PanMechanics_2025` | 23 | 874.87 | 667 | 1150 |
| `PanPhO_2024` | 33 | 2820.55 | 1448 | 3880 |
| `PanPhO_2025` | 47 | 3526.47 | 1561 | 6209 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 413 |
| Images per Sample | min: 1, max: 5, mean: 1.5 |
| Resolution Range | 456x60 - 3200x1645 |
| Formats | png |


## Sample Example

**Subset**: `APhO_2025`

```json
{
  "input": [
    {
      "id": "9ec40cee",
      "content": [
        {
          "text": "You are participating in a high school physics Olympiad exam.\nPlease read the following question carefully and provide a clear, step-by-step solution with full reasoning.\nInstructions:\n1. Use LaTeX to format all variables, equations, and calc ... [TRUNCATED 3334 chars] ... gamma} R^{\\delta}$ \nwhere $G$ is the gravitational constant, and $\\beta, \\gamma$ and $\\delta$ are constant exponents.\nQuestion (Answer only the question stated below):\nFind the values of exponents: (1) $\\beta$, (2) $\\gamma$, and (3) $\\delta$."
        },
        {
          "image": "[BASE64_IMAGE: png, ~101.8KB]"
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "APhO_2025",
  "metadata": {
    "id": "APhO_2025_1_A_1",
    "source": "APhO_2025",
    "question": "Find the values of exponents: (1) $\\beta$, (2) $\\gamma$, and (3) $\\delta$.",
    "answers": [
      "\\boxed{$\\beta = 2$}",
      "\\boxed{$\\gamma = -1$}",
      "\\boxed{$\\delta = 4$}"
    ],
    "marking": [
      [
        "Award 0.2 pt if the answer correctly expresses the dimension of $G$ as $[G] = L^3 M^{-1} T^{-2}$, where $L$ is the base dimensions length, $M$ is mass, and $T$ is time. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer correctly sets up the exponent equation $0 = 2 - \\beta$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer correctly sets up the exponent equation $0 = \\gamma + 1$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer correctly sets up the exponent equation $1 = \\delta - 3$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer obtains the correct value $\\beta = 2$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer obtains the correct value $\\gamma = -1$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer obtains the correct value $\\delta = 4$. Otherwise, award 0 pt."
      ]
    ]
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hipho \
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
    datasets=['hipho'],
    dataset_args={
        'hipho': {
            # subset_list: ['APhO_2025', 'CPhO_2025', 'EuPhO_2024']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
