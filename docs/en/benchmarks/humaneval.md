# HumanEval


## Overview

HumanEval is a benchmark for evaluating the code generation capabilities of language models. It consists of 164 hand-written Python programming problems with function signatures, docstrings, and comprehensive test cases.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Function signature with docstring describing the expected behavior
- **Output**: Complete Python function implementation
- **Languages**: Python only

## Key Features

- 164 hand-crafted programming problems
- Each problem includes a function signature, docstring, and test cases
- Problems range from simple string manipulation to complex algorithms
- Canonical solutions provided for reference
- Automatic correctness verification through test execution

## Evaluation Notes

- **Security Warning**: By default, code is executed in the local environment. We strongly recommend using sandbox execution for safety. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Supports `pass@k` metric calculation for measuring generation quality
- Default timeout is 4 seconds per problem
- Code is extracted from markdown code blocks if present


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `humaneval` |
| **Dataset ID** | [opencompass/humaneval](https://modelscope.cn/datasets/opencompass/humaneval/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_at_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 164 |
| Prompt Length (Mean) | 609.6 chars |
| Prompt Length (Min/Max) | 274 / 1519 chars |

## Sample Example

**Subset**: `openai_humaneval`

```json
{
  "input": [
    {
      "id": "5f652252",
      "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: f ... [TRUNCATED] ... Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
    }
  ],
  "target": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "HumanEval/0",
    "entry_point": "has_close_elements",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0 ... [TRUNCATED] ...  candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{question}
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
    --datasets humaneval \
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
    datasets=['humaneval'],
    use_sandbox=True,
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


