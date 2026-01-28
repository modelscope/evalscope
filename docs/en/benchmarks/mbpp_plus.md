# MBPP-Plus


## Overview

MBPP Plus is a fortified version of the MBPP benchmark, created to improve evaluation reliability for basic Python programming synthesis. It addresses quality issues in the original dataset and significantly increases test coverage for each problem.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Programming task description with test assertions
- **Output**: Python function implementation
- **Improvements**: Sanitized data with expanded test suites

## Key Features

- Corrected reference code and clarified problem statements
- Massively expanded test cases per problem (automated generation)
- LLM-based and mutation-based test generation strategies
- Exposes edge-case bugs missed by original MBPP
- More rigorous and accurate code correctness evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- **Security Warning**: By default, code is executed in the local environment. We strongly recommend using sandbox execution. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Supports `pass@k` metric calculation
- Default timeout is 20 seconds per problem
- Stricter evaluation than original MBPP benchmark


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mbpp_plus` |
| **Dataset ID** | [evalscope/mbppplus](https://modelscope.cn/datasets/evalscope/mbppplus/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_at_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 378 |
| Prompt Length (Mean) | 375.53 chars |
| Prompt Length (Min/Max) | 222 / 2801 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "aa126494",
      "content": "You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists. Your code should pass these tests:\n\nassert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\nassert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))\nassert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))"
    }
  ],
  "target": "\ndef similar_elements(test_tup1, test_tup2):\n  return tuple(set(test_tup1) & set(test_tup2))\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": 2,
    "source_file": "Benchmark Questions Verification V2.ipynb",
    "test_imports": [],
    "test_list": [
      "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
      "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
      "assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))"
    ],
    "test": "import numpy as np\nfrom math import inf\n\ndef is_floats(x) -> bool:\n    # check if it is float; List[float]; Tuple[float]\n    if isinstance(x, float):\n        return True\n    if isinstance(x, (list, tuple)):\n        return all(isinstance(i, fl ... [TRUNCATED] ... wvS', 'tUqF', 'ITntCqEvPi', 'kx'), (1, 2, 3, 4, 5, 6, 7, 8, 9), (11, 12, 13, 14, 15, 16, 17, 19, 20), (19, 5), (1, 2, 3, 6, 7, 8, 9, 10, 12)]\nfor i, (inp, exp) in enumerate(zip(inputs, results)):\n    assertion(similar_elements(*inp), exp, 0)\n"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
```

## Sandbox Configuration

This benchmark requires a sandbox environment for code execution.

```json
{
  "image": "python:3.11-slim",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  }
}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mbpp_plus \
    --use-sandbox \
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
    datasets=['mbpp_plus'],
    use_sandbox=True,
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


