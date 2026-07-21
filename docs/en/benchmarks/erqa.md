# ERQA


## Overview

ERQA (Embodied Reasoning QA) is a benchmark for evaluating spatial reasoning and embodied understanding capabilities of multimodal large language models. It tests models' ability to reason about trajectories, actions, spatial relationships, and task planning in egocentric robotic scenarios.

## Task Description

- **Task Type**: Embodied Spatial Reasoning (Multiple Choice)
- **Input**: Egocentric image(s) + multiple-choice question (A/B/C/D)
- **Output**: Single answer letter (A/B/C/D)
- **Domain**: Robotics, spatial reasoning, embodied AI

## Key Features

- 400 questions across 8 reasoning categories
- Multi-image support (some questions require reasoning across multiple views)
- Categories: Trajectory Reasoning, Action Reasoning, Pointing, State Estimation, Spatial Reasoning, Multi-view Reasoning, Task Reasoning, Other
- Egocentric perspective from robotic manipulation scenarios

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Answers are single letters (A/B/C/D)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `erqa` |
| **Dataset ID** | [evalscope/ERQA](https://modelscope.cn/datasets/evalscope/ERQA/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `MultiModal`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 400 |
| Prompt Length (Mean) | 289.05 chars |
| Prompt Length (Min/Max) | 147 / 834 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Trajectory Reasoning` | 66 | 307.3 | 161 | 834 |
| `Action Reasoning` | 72 | 271.97 | 175 | 503 |
| `Pointing` | 34 | 300.82 | 178 | 676 |
| `State Estimation` | 55 | 314.84 | 147 | 735 |
| `Spatial Reasoning` | 84 | 286.21 | 152 | 819 |
| `Multi-view Reasoning` | 37 | 277.78 | 196 | 475 |
| `Task Reasoning` | 38 | 280.18 | 152 | 737 |
| `Other` | 14 | 231.79 | 188 | 409 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 630 |
| Images per Sample | min: 1, max: 16, mean: 1.57 |
| Resolution Range | 200x150 - 3072x4080 |
| Formats | jpeg |


## Sample Example

**Subset**: `Trajectory Reasoning`

```json
{
  "input": [
    {
      "id": "eec4c5cf",
      "content": [
        {
          "text": "If the yellow robot gripper follows the yellow trajectory, what will happen? Choices: A. Robot puts the soda on the wooden steps. B. Robot moves the soda in front of the wooden steps. C. Robot moves the soda to the very top of the wooden steps. D. Robot picks up the soda can and moves it up. Please answer directly with only the letter of the correct option and nothing else."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~16.0KB]"
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
  "subset_key": "Trajectory Reasoning",
  "metadata": {
    "question_id": "ERQA_1",
    "question_type": "Trajectory Reasoning"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets erqa \
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
    datasets=['erqa'],
    dataset_args={
        'erqa': {
            # subset_list: ['Trajectory Reasoning', 'Action Reasoning', 'Pointing']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
