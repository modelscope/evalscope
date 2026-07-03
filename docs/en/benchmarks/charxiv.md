# CharXiv


## Overview

CharXiv is a comprehensive chart understanding benchmark from NeurIPS 2024 that evaluates multimodal
large language models on realistic scientific charts from arXiv papers. It tests both low-level
chart element perception (descriptive) and high-level reasoning about chart data.

## Task Description

- **Task Type**: Chart Understanding (Descriptive + Reasoning)
- **Input**: Scientific chart image + question
- **Output**: Free-form text answer
- **Domains**: cs, physics, math, eess, q-bio, q-fin, stat, econ

## Key Features

- 2,323 real scientific charts from arXiv papers across 8 disciplines
- Two question types:
  - **Descriptive** (4 per chart): Basic element identification (titles, axes, legends, trends, etc.)
  - **Reasoning** (1 per chart): Higher-order reasoning requiring data synthesis
- 19 descriptive question templates covering information extraction, enumeration, pattern recognition, counting, and compositionality
- 4 reasoning answer types: text-in-chart, text-in-general, number-in-chart, number-in-general
- Validation set (1,000 charts) and test set (1,323 charts)
- Evaluation via LLM judge following the official CharXiv grading protocol

## Evaluation Notes

- Default evaluation uses the **validation** split (1,000 charts, 5,000 questions)
- Each chart yields 5 samples: 4 descriptive + 1 reasoning
- Primary metric: **Accuracy** via LLM-as-judge
- Subsets: `descriptive` and `reasoning` (also by category)
- Requires `judge_model_args` configuration for LLM judge
- [Paper](https://arxiv.org/abs/2406.18521) | [GitHub](https://github.com/princeton-nlp/CharXiv)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `charxiv` |
| **Dataset ID** | [princeton-nlp/CharXiv](https://modelscope.cn/datasets/princeton-nlp/CharXiv/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2406.18521) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 5,000 |
| Prompt Length (Mean) | 276.24 chars |
| Prompt Length (Min/Max) | 80 / 687 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `descriptive` | 4,000 | 261.51 | 156 | 432 |
| `reasoning` | 1,000 | 335.14 | 80 | 687 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 5,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 1023x139 - 1024x1024 |
| Formats | jpeg |


## Sample Example

**Subset**: `descriptive`

```json
{
  "input": [
    {
      "id": "44ff5b8a",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~70.0KB]"
        },
        {
          "text": "For the current plot, what is the spatially highest labeled tick on the y-axis?\n* Your final answer should be the tick value on the y-axis that is explicitly written. Ignore units or scales that are written separately from the tick."
        }
      ]
    }
  ],
  "target": "60",
  "id": 0,
  "group_id": 0,
  "subset_key": "descriptive",
  "metadata": {
    "question_type": "descriptive",
    "question_id": 7,
    "category": "cs",
    "original_id": "2004.10956"
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
    --datasets charxiv \
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
    datasets=['charxiv'],
    dataset_args={
        'charxiv': {
            # subset_list: ['descriptive', 'reasoning']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


