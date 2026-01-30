# MATH-500


## Overview

MATH-500 is a curated subset of 500 problems from the MATH benchmark, designed to evaluate the mathematical reasoning capabilities of language models. It covers five difficulty levels across various mathematical topics including algebra, geometry, number theory, and calculus.

## Task Description

- **Task Type**: Mathematical Problem Solving
- **Input**: Mathematical problem statement
- **Output**: Step-by-step solution with final numerical answer
- **Difficulty Levels**: Level 1 (easiest) to Level 5 (hardest)

## Key Features

- 500 carefully selected problems from the full MATH dataset
- Five difficulty levels for fine-grained evaluation
- Problems cover algebra, geometry, number theory, probability, and more
- Each problem includes a reference solution
- Designed for efficient yet comprehensive math evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Numeric equivalence checking for answer comparison
- Results can be broken down by difficulty level
- Commonly used for math reasoning benchmarking due to manageable size


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `math_500` |
| **Dataset ID** | [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 500 |
| Prompt Length (Mean) | 266.89 chars |
| Prompt Length (Min/Max) | 91 / 1804 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Level 1` | 43 | 193.19 | 100 | 571 |
| `Level 2` | 90 | 218.82 | 91 | 802 |
| `Level 3` | 105 | 236 | 93 | 688 |
| `Level 4` | 128 | 277.1 | 93 | 1771 |
| `Level 5` | 134 | 337.28 | 118 | 1804 |

## Sample Example

**Subset**: `Level 1`

```json
{
  "input": [
    {
      "id": "b5d90091",
      "content": "Suppose $\\sin D = 0.7$ in the diagram below. What is $DE$? [asy]\npair D,E,F;\nF = (0,0);\nD = (sqrt(51),7);\nE = (0,7);\ndraw(D--E--F--D);\ndraw(rightanglemark(D,E,F,15));\nlabel(\"$D$\",D,NE);\nlabel(\"$E$\",E,NW);\nlabel(\"$F$\",F,SW);\nlabel(\"$7$\",(E+F)/2,W);\n[/asy]\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "\\sqrt{51}",
  "id": 0,
  "group_id": 0,
  "subset_key": "Level 1",
  "metadata": {
    "question_id": "test/precalculus/1303.json",
    "solution": "The triangle is a right triangle, so $\\sin D = \\frac{EF}{DF}$. Then we have that $\\sin D = 0.7 = \\frac{7}{DF}$, so $DF = 10$.\n\nUsing the Pythagorean Theorem, we find that the length of $DE$ is $\\sqrt{DF^2 - EF^2},$ or $\\sqrt{100 - 49} = \\boxed{\\sqrt{51}}$."
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
    --datasets math_500 \
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
    datasets=['math_500'],
    dataset_args={
        'math_500': {
            # subset_list: ['Level 1', 'Level 2', 'Level 3']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


