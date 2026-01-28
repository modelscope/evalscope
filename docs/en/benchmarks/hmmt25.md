# HMMT25


## Overview

HMMT February 2025 (MathArena) is a challenging evaluation benchmark derived from the Harvard-MIT Mathematics Tournament (HMMT) February 2025 competition, one of the most prestigious and difficult high school math contests globally.

## Task Description

- **Task Type**: Competition Mathematics Problem Solving
- **Input**: HMMT-level mathematical problem
- **Output**: Answer with step-by-step reasoning
- **Difficulty**: Advanced high school competition level

## Key Features

- Problems from HMMT February 2025 competition
- Four primary domains: Algebra, Combinatorics, Geometry, Number Theory
- Highly challenging competition-level problems
- Tests advanced mathematical reasoning
- Represents elite high school mathematics difficulty

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Numeric accuracy metric
- Problems span multiple mathematical domains


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hmmt25` |
| **Dataset ID** | [evalscope/hmmt_feb_2025](https://modelscope.cn/datasets/evalscope/hmmt_feb_2025/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 30 |
| Prompt Length (Mean) | 410.53 chars |
| Prompt Length (Min/Max) | 173 / 1594 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "fb6fb4e8",
      "content": "Problem:\nCompute the sum of the positive divisors (including $1$) of $9!$ that have units digit $1$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "103",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_idx": 1,
    "problem_type": [
      "Number Theory"
    ]
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hmmt25 \
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
    datasets=['hmmt25'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


