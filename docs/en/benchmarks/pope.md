# POPE


## Overview

POPE (Polling-based Object Probing Evaluation) is a benchmark specifically designed to evaluate object hallucination in Large Vision-Language Models (LVLMs). It tests models' ability to accurately identify objects present in images through yes/no questions.

## Task Description

- **Task Type**: Object Hallucination Detection (Yes/No Q&A)
- **Input**: Image with question "Is there a [object] in the image?"
- **Output**: YES or NO answer
- **Focus**: Measuring accuracy vs. hallucination rate

## Key Features

- Three sampling strategies: random, popular, adversarial
- Tests for false positive object claims (hallucination)
- Based on MSCOCO images
- Simple yes/no question format for objective evaluation
- Measures alignment between model responses and visual content

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Five metrics: accuracy, precision, recall, F1 score, yes_ratio
- F1 score is the primary aggregation metric
- Three subsets: `popular`, `adversarial`, `random`
- "Popular" and "adversarial" subsets are more challenging
- yes_ratio indicates model's tendency to answer "yes"


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `pope` |
| **Dataset ID** | [lmms-lab/POPE](https://modelscope.cn/datasets/lmms-lab/POPE/summary) |
| **Paper** | N/A |
| **Tags** | `Hallucination`, `MultiModal`, `Yes/No` |
| **Metrics** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `N/A` |
| **Aggregation** | `f1` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 9,000 |
| Prompt Length (Mean) | 79.4 chars |
| Prompt Length (Min/Max) | 75 / 87 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `popular` | 3,000 | 79.27 | 75 | 87 |
| `adversarial` | 3,000 | 79.36 | 75 | 87 |
| `random` | 3,000 | 79.59 | 75 | 87 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 9,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 500x243 - 640x640 |
| Formats | jpeg |


## Sample Example

**Subset**: `popular`

```json
{
  "input": [
    {
      "id": "8847a5a3",
      "content": [
        {
          "text": "Is there a snowboard in the image?\nPlease answer YES or NO without an explanation."
        },
        {
          "image": "[BASE64_IMAGE: png, ~87.2KB]"
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "3000",
    "answer": "YES",
    "category": "popular",
    "question_id": "1"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
Please answer YES or NO without an explanation.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets pope \
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
    datasets=['pope'],
    dataset_args={
        'pope': {
            # subset_list: ['popular', 'adversarial', 'random']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


