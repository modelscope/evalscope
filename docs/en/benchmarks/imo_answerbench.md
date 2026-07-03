# IMO-AnswerBench

## Overview

IMO-AnswerBench is a benchmark of 400 challenging problems sourced from the International Mathematical Olympiad (IMO) Shortlists. It covers four major mathematical domains and is designed to evaluate advanced mathematical reasoning capabilities of language models at the olympiad level.

## Task Description

- **Task Type**: Olympiad Mathematics Problem Solving
- **Input**: IMO Shortlist mathematical problem
- **Output**: Step-by-step solution with final answer
- **Difficulty**: International olympiad level

## Key Features

- 400 problems from IMO Shortlists (2005-2024)
- Four domains: Algebra, Combinatorics, Geometry, Number Theory
- Subcategories include: Operation, Inequality, Sequence, Polynomial, Functional Equation, etc.
- Answers range from simple integers to complex LaTeX expressions (intervals, sets, fractions)
- Represents the highest difficulty level in mathematical problem-solving benchmarks

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Uses numeric equivalence checking with LLM-as-judge for complex answers
- Results can be broken down by Category (Algebra, Combinatorics, Geometry, Number Theory)
- Many answers involve symbolic expressions requiring mathematical equivalence checking

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `imo_answerbench` |
| **Dataset ID** | [evalscope/imo-answerbench](https://modelscope.cn/datasets/evalscope/imo-answerbench/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 400 |
| Prompt Length (Mean) | 423.02 chars |
| Prompt Length (Min/Max) | 145 / 1343 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Algebra` | 100 | 345.06 | 183 | 984 |
| `Combinatorics` | 100 | 641.53 | 257 | 1343 |
| `Geometry` | 100 | 375.44 | 196 | 703 |
| `Number theory` | 100 | 330.07 | 145 | 769 |

## Sample Example

**Subset**: `Algebra`

```json
{
  "input": [
    {
      "id": "afe74111",
      "content": "Problem:\nFor a given positive integer $N$, Henry writes the quotient of $ab$ divided by $N+1$ on the board for each integer pair $(a,b)$ where $1\\le a,b\\le N$. Find all $N$ such that the sum of the $N^2$ numbers Henry wrote on the board is $\\frac{N^3-N^2+2}{4}$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "3",
  "id": 0,
  "group_id": 0,
  "subset_key": "Algebra",
  "metadata": {
    "problem_id": "imo-bench-algebra-001",
    "subcategory": "Operation",
    "source": "IMO Shortlist 2021"
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
    --datasets imo_answerbench \
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
    datasets=['imo_answerbench'],
    dataset_args={
        'imo_answerbench': {
            # subset_list: ['Algebra', 'Combinatorics', 'Geometry']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


