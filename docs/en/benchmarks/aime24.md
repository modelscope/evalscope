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
- Reference solutions available in metadata for analysis


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `aime24` |
| **Dataset ID** | [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 30 |
| Prompt Length (Mean) | 405.33 chars |
| Prompt Length (Min/Max) | 185 / 1009 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "2f896eed",
      "content": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she wa ... [TRUNCATED] ... e coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "204",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_id": 60,
    "solution": "$\\frac{9}{s} + t = 4$ in hours and $\\frac{9}{s+2} + t = 2.4$ in hours.\nSubtracting the second equation from the first, we get, \n$\\frac{9}{s} - \\frac{9}{s+2} = 1.6$\nMultiplying by $(s)(s+2)$, we get \n$9s+18-9s=18=1.6s^{2} + 3.2s$\nMultiplying b ... [TRUNCATED] ... s = 2.5$. Now, $2.5+0.5 = 3$. Taking $\\frac{9}{3} = 3$, we find that it will take three hours for the 9 kilometers to be traveled. The t minutes spent at the coffeeshop can be written as $144-48(2.5)$, so t = 24. $180 + 24 = 204$. -sepehr2010"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
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


