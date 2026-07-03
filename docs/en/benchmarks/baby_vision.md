# BabyVision


## Overview

BabyVision is a visual perception benchmark that evaluates the fundamental visual abilities of
multimodal large language models through tasks inspired by infant and early childhood visual
development. It focuses on fine-grained discrimination, spatial perception, visual pattern
recognition, and visual tracking.

## Task Description

- **Task Type**: Visual Perception (Choice + Fill-in-the-blank)
- **Input**: Image + question
- **Output**: Choice letter or free-form short answer
- **Domains**: Fine-grained discrimination, spatial perception, visual pattern recognition, visual tracking

## Key Features

- 388 test samples across 4 major visual ability categories and 22 subtypes
- Two answer types: choice (135 samples) and blank (253 samples)
- Subtypes include: Find the different, Find the same, Count clusters, Maze,
  3D cube unfold, Pattern completion, Paper folding, Rotation patterns, etc.
- Tests low-level visual perception rather than high-level reasoning or knowledge
- Includes Chain-of-Thought (CoT) reference for analysis

## Evaluation Notes

- Default evaluation uses the **train** split (388 samples, single split dataset)
- Primary metric: **Accuracy** via LLM-as-judge
- Subsets organized by `type` field (4 categories)
- LLM judge evaluates both choice and blank answer types uniformly
- Requires `judge_model_args` configuration for LLM judge


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `baby_vision` |
| **Dataset ID** | [evalscope/BabyVision](https://modelscope.cn/datasets/evalscope/BabyVision/summary) |
| **Paper** | N/A |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 388 |
| Prompt Length (Mean) | 167.37 chars |
| Prompt Length (Min/Max) | 33 / 450 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Fine-grained Discrimination` | 163 | 152.09 | 33 | 450 |
| `Spatial Perception` | 91 | 157.18 | 73 | 370 |
| `Visual Pattern Recognition` | 51 | 178.92 | 94 | 319 |
| `Visual Tracking` | 83 | 201.45 | 97 | 389 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 388 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 174x144 - 2378x1448 |
| Formats | jpeg, png, webp |


## Sample Example

**Subset**: `Fine-grained Discrimination`

```json
{
  "input": [
    {
      "id": "8b83904b",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~77.0KB]"
        },
        {
          "text": "The image shows a total of 49 tiger patterns arranged in 7 rows and 7 columns. One of them is different from the others. Which row and column is it in? The answer format is (x,y). (For example, the answer for the 2nd row and 3rd column is (2,3))."
        }
      ]
    }
  ],
  "target": "(4,7)",
  "id": 0,
  "group_id": 0,
  "subset_key": "Fine-grained Discrimination",
  "metadata": {
    "taskId": 445,
    "type": "Fine-grained Discrimination",
    "subtype": "Find the different",
    "ansType": "blank",
    "coT": "The image shows 49 tiger patterns arranged in 7 rows and 7 columns.\nNow, we need to find the coordinates of the one tiger pattern that is different from the other 48.\nIt can be observed that the tiger in the fourth row and seventh column has no ears (the ears are located in the upper right corner of each tiger pattern), while the other 48 tigers have ears.\nTherefore, the correct answer is (4,7)."
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
    --datasets baby_vision \
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
    datasets=['baby_vision'],
    dataset_args={
        'baby_vision': {
            # subset_list: ['Fine-grained Discrimination', 'Spatial Perception', 'Visual Pattern Recognition']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


