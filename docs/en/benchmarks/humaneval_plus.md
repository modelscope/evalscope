# HumanEvalPlus


## Overview

HumanEval Plus is a rigorous extension of OpenAI's HumanEval benchmark, designed to address high false-positive rates in code generation evaluation. It augments the original test cases with tens of thousands of automatically generated inputs to expose edge-case bugs and functional errors.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Function signature with docstring describing expected behavior
- **Output**: Complete Python function implementation
- **Test Coverage**: Massively expanded compared to original HumanEval

## Key Features

- 164 problems from original HumanEval with enhanced test suites
- Tens of thousands of additional test cases per problem
- LLM-based and mutation-based test input generation
- Exposes edge cases and bugs missed by original HumanEval
- Much stricter and more accurate correctness evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- **Security Warning**: By default, code is executed in the local environment. We strongly recommend using sandbox execution. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Supports `pass@k` metric calculation
- Default timeout is 300 seconds to accommodate extensive test suites
- Uses custom Docker image with numpy pre-installed


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `humaneval_plus` |
| **Dataset ID** | [evalscope/humanevalplus](https://modelscope.cn/datasets/evalscope/humanevalplus/summary) |
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
| Prompt Length (Mean) | 609.57 chars |
| Prompt Length (Min/Max) | 274 / 1519 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "dcff8346",
      "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: f ... [TRUNCATED] ... Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
    }
  ],
  "target": "\n\n    sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "HumanEval/0",
    "entry_point": "has_close_elements",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "test": "\n\nimport numpy as np\n\ndef is_floats(x) -> bool:\n    # check if it is float; List[float]; Tuple[float]\n    if isinstance(x, float):\n        return True\n    if isinstance(x, (list, tuple)):\n        return all(isinstance(i, float) for i in x)\n   ... [TRUNCATED] ...  True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True]\n    for i, (inp, exp) in enumerate(zip(inputs, results)):\n        assertion(candidate(*inp), exp, 0)\n"
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
  "image": "python3.11-numpy",
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
    --datasets humaneval_plus \
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
    datasets=['humaneval_plus'],
    use_sandbox=True,
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


