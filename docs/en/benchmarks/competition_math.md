# Competition-MATH


## Overview

Competition-MATH is a comprehensive benchmark of 12,500 challenging competition mathematics problems collected from AMC, AIME, and other prestigious math competitions. It is designed to evaluate the advanced mathematical reasoning capabilities of language models.

## Task Description

- **Task Type**: Competition Mathematics Problem Solving
- **Input**: Mathematical problem from competitions
- **Output**: Step-by-step solution with final answer in \boxed{}
- **Difficulty Levels**: Level 1 (easiest) to Level 5 (hardest)

## Key Features

- 12,500 problems from mathematics competitions
- Five difficulty levels for comprehensive evaluation
- Topics: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus
- Each problem includes human-written solutions
- Designed for evaluating advanced mathematical reasoning

## Evaluation Notes

- Default configuration uses **4-shot** examples with Chain-of-Thought prompting
- Answers are extracted from `\boxed{}` format
- Numeric equivalence checking for answer comparison
- Results can be analyzed by level and problem type
- Uses math_parser for robust answer extraction


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `competition_math` |
| **Dataset ID** | [evalscope/competition_math](https://modelscope.cn/datasets/evalscope/competition_math/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 4-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 5,000 |
| Prompt Length (Mean) | 1019.03 chars |
| Prompt Length (Min/Max) | 675 / 3991 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Level 1` | 437 | 947.84 | 842 | 2757 |
| `Level 2` | 894 | 816.87 | 682 | 2328 |
| `Level 3` | 1,131 | 822.14 | 675 | 2470 |
| `Level 4` | 1,214 | 949.04 | 768 | 2871 |
| `Level 5` | 1,324 | 1411.41 | 1187 | 3991 |

## Sample Example

**Subset**: `Level 1`

```json
{
  "input": [
    {
      "id": "3848cf6a",
      "content": "Here are some examples of how to solve similar problems:\n\nProblem:\nWhen Joyce counts the pennies in her bank by fives, she has one left over. When she counts them by threes, there are two left over. What is the least possible number of pennie ... [TRUNCATED] ... of 12, how many eggs will be left over if all cartons are sold?\nSolution:\n4\nProblem:\nHow many of the six integers 1 through 6 are divisors of the four-digit number 1452?\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "5",
  "id": 0,
  "group_id": 0,
  "subset_key": "Level 1",
  "metadata": {
    "reasoning": "All numbers are divisible by $1$. The last two digits, $52$, form a multiple of 4, so the number is divisible by $4$, and thus $2$. $1+4+5+2=12$, which is a multiple of $3$, so $1452$ is divisible by $3$.  Since it is divisible by $2$ and $3$, it is divisible by $6$.  But it is not divisible by $5$ as it does not end in $5$ or $0$.  So the total is $\\boxed{5}$.",
    "type": "Number Theory"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

<details>
<summary>Few-shot Template</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets competition_math \
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
    datasets=['competition_math'],
    dataset_args={
        'competition_math': {
            # subset_list: ['Level 1', 'Level 2', 'Level 3']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


