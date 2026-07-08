# AIME-2025


## Overview

AIME 2025 (American Invitational Mathematics Examination 2025) is a benchmark based on problems from the prestigious AIME competition, one of the most challenging high school mathematics contests in the United States. It tests advanced mathematical reasoning and problem-solving skills.

## Task Description

- **Task Type**: Competition Mathematics Problem Solving
- **Input**: AIME-level mathematical problem
- **Output**: Integer answer (0-999) with step-by-step reasoning
- **Difficulty**: Advanced high school / early undergraduate level

## Key Features

- Problems from AIME I and AIME II 2025 competitions
- Answers are always integers between 0 and 999
- Requires creative mathematical reasoning and problem-solving
- Topics: algebra, geometry, number theory, combinatorics, probability
- Represents top-tier high school mathematics competition difficulty

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Uses LLM-as-judge for mathematical equivalence checking


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `aime25` |
| **Dataset ID** | [evalscope/aime25](https://modelscope.cn/datasets/evalscope/aime25/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 30 |
| Prompt Length (Mean) | 604.93 chars |
| Prompt Length (Min/Max) | 208 / 1862 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "bff66863",
      "content": "\nSolve the following math problem step by step. Put your answer inside \\boxed{}.\n\nFind the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$\n\nRemember to put your answer inside \\boxed{}."
    }
  ],
  "target": "70",
  "id": 0,
  "group_id": 0
}
```

## Prompt Template

**Prompt Template:**
```text

Solve the following math problem step by step. Put your answer inside \boxed{{}}.

{question}

Remember to put your answer inside \boxed{{}}.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets aime25 \
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
    datasets=['aime25'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
