# AMC


## Overview

AMC (American Mathematics Competitions) is a benchmark based on problems from the AMC 10/12 competitions from 2022-2024. These multiple-choice problems test mathematical problem-solving skills at the high school level and serve as qualifiers for the AIME competition.

## Task Description

- **Task Type**: Competition Mathematics (Multiple Choice)
- **Input**: AMC-level mathematical problem
- **Output**: Correct answer with step-by-step reasoning
- **Years Covered**: 2022, 2023, 2024

## Key Features

- Problems from AMC 10 and AMC 12 competitions (2022-2024)
- Multiple-choice format with 5 answer options
- Topics: algebra, geometry, number theory, combinatorics
- Difficulty ranges from accessible to challenging
- Official competition problems with verified solutions

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Three subsets available: `amc22`, `amc23`, `amc24`
- Problems include original URLs for reference
- Solutions available in metadata for verification


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `amc` |
| **Dataset ID** | [evalscope/amc_22-24](https://modelscope.cn/datasets/evalscope/amc_22-24/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `N/A` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 134 |
| Prompt Length (Mean) | 324.58 chars |
| Prompt Length (Min/Max) | 98 / 1218 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `amc22` | 43 | 337.42 | 129 | 934 |
| `amc23` | 46 | 337.98 | 143 | 1218 |
| `amc24` | 45 | 298.62 | 98 | 882 |

## Sample Example

**Subset**: `amc22`

```json
{
  "input": [
    {
      "id": "54851a8f",
      "content": "What is the value of\\[3+\\frac{1}{3+\\frac{1}{3+\\frac13}}?\\]\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "\\frac{109}{33}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "year": 2022,
    "url": "https://artofproblemsolving.com/wiki/index.php/2022_AMC_12A_Problems/Problem_1",
    "solution": "We have\\begin{align*} 3+\\frac{1}{3+\\frac{1}{3+\\frac13}} &= 3+\\frac{1}{3+\\frac{1}{\\left(\\frac{10}{3}\\right)}} \\\\ &= 3+\\frac{1}{3+\\frac{3}{10}} \\\\ &= 3+\\frac{1}{\\left(\\frac{33}{10}\\right)} \\\\ &= 3+\\frac{10}{33} \\\\ &= \\boxed{\\textbf{(D)}\\ \\frac{109}{33}}. \\end{align*}"
  }
}
```

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
    --datasets amc \
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
    datasets=['amc'],
    dataset_args={
        'amc': {
            # subset_list: ['amc22', 'amc23', 'amc24']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


