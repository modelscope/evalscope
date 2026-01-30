# VisuLogic


## Overview

VisuLogic is a benchmark for evaluating visual reasoning capabilities of Multimodal Large Language Models (MLLMs), independent of textual reasoning. It features carefully constructed visual reasoning tasks that are inherently difficult to articulate using language alone.

## Task Description

- **Task Type**: Visual Reasoning (Multiple-Choice)
- **Input**: Image + visual reasoning question with 4 choices
- **Output**: Answer letter (A/B/C/D)
- **Domains**: Pure visual reasoning without text-based shortcuts

## Key Features

- Six reasoning skill categories:
  - **Quantitative Reasoning**: Understanding quantity changes in images
  - **Positional Reasoning**: Understanding spatial positions
  - **Spatial Reasoning**: Understanding 3D spatial relationships
  - **Attribute Reasoning**: Understanding visual attributes
  - **Stylistic Reasoning**: Understanding visual styles
  - **Other**: Miscellaneous visual reasoning tasks
- Tests genuine visual understanding beyond language shortcuts

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting with "ANSWER: [LETTER]" format
- Results grouped by reasoning skill category


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `visulogic` |
| **Dataset ID** | [evalscope/VisuLogic](https://modelscope.cn/datasets/evalscope/VisuLogic/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 394.16 chars |
| Prompt Length (Min/Max) | 285 / 697 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Quantitative Reasoning` | 353 | 399.15 | 308 | 697 |
| `Other` | 108 | 399.08 | 285 | 560 |
| `Positional Reasoning` | 136 | 372.37 | 295 | 448 |
| `Stylistic Reasoning` | 90 | 375.32 | 303 | 483 |
| `Spatial Reasoning` | 231 | 401.91 | 314 | 537 |
| `Attribute Reasoning` | 82 | 401.15 | 295 | 458 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 288x125 - 700x825 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `Quantitative Reasoning`

```json
{
  "input": [
    {
      "id": "1a6407d8",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.\n\nFrom the four given options, select the most suitable one to fill in the question mark, so that a certain regularity is presented:\n\n\n\nA: A  \nB: B  \nC: C  \nD: D"
        },
        {
          "image": "[BASE64_IMAGE: png, ~44.9KB]"
        }
      ]
    }
  ],
  "choices": [
    "A",
    "B",
    "C",
    "D"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "Quantitative Reasoning",
  "metadata": {
    "id": "00000"
  }
}
```

## Prompt Template

**Prompt Template:**
```text

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets visulogic \
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
    datasets=['visulogic'],
    dataset_args={
        'visulogic': {
            # subset_list: ['Quantitative Reasoning', 'Other', 'Positional Reasoning']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


