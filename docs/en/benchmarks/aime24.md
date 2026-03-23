# AIME-2024


## Overview

AIME 2024 (American Invitational Mathematics Examination 2024) is a benchmark based on problems from the prestigious AIME competition. These problems represent some of the most challenging high school mathematics problems, requiring creative problem-solving and advanced mathematical reasoning.

## Task Description

- **Task Type**: Competition Mathematics Problem Solving
- **Input**: AIME-level mathematical problem
- **Output**: Integer answer (0-999) with step-by-step reasoning
- **Difficulty**: Advanced high school / early undergraduate level

## Key Features

- Problems from the 2024 AIME I and AIME II competitions
- Answers are always integers between 0 and 999
- Requires creative mathematical reasoning and problem-solving
- Topics: algebra, geometry, number theory, combinatorics, probability
- Represents top-tier high school mathematics competition difficulty

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Only integer answers are accepted (matching AIME format)
- Problems are significantly harder than GSM8K or standard MATH benchmark


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `aime24` |
| **Dataset ID** | [evalscope/aime24](https://modelscope.cn/datasets/evalscope/aime24/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 30 |
| Prompt Length (Mean) | 462.33 chars |
| Prompt Length (Min/Max) | 242 / 1066 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "32187d6f",
      "content": "\nSolve the following math problem step by step. Put your answer inside \\boxed{}.\n\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the w ... [TRUNCATED 164 chars] ... g $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n\nRemember to put your answer inside \\boxed{}."
    }
  ],
  "target": "\\boxed{204}",
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
    --datasets aime24 \
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
    datasets=['aime24'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


