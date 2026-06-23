# BigCodeBench


## Overview

BigCodeBench is an easy-to-use benchmark for solving practical and challenging tasks via code. It evaluates the true programming capabilities of large language models (LLMs) in a more realistic setting with diverse function calls from 139 popular libraries covering 723 API calls.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Programming task description (docstring or natural language instruction)
- **Output**: Complete Python function implementation
- **Libraries**: 139 popular Python libraries (numpy, pandas, sklearn, etc.)

## Key Features

- 1,140 rich-context programming tasks in Python
- Two evaluation modes: Complete (docstring) and Instruct (natural language)
- Covers diverse function calls from 139 popular libraries
- Uses unittest.TestCase for thorough correctness verification
- Supports pass@k metric calculation

## Evaluation Notes

- **Sandbox Required**: Requires sandbox environment with 70+ Python libraries pre-installed
- Two modes available via `split` parameter: `complete` (docstring completion) or `instruct` (NL instruction)
- Default timeout is 240 seconds per problem
- `calibrate` option prepends code_prompt to align function signatures
- See [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for setup


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bigcodebench` |
| **Dataset ID** | [evalscope/bigcodebench](https://modelscope.cn/datasets/evalscope/bigcodebench/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `v0.1.4` |
| **Aggregation** | `mean_and_pass_at_k` |


## Data Statistics

*Statistics not available.*

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "657d2473",
      "content": "Calculates the average of the sums of absolute differences between each pair of consecutive numbers for all permutations of a given list. Each permutation is shuffled before calculating the differences. Args: - numbers (list): A list of numbe ... [TRUNCATED 75 chars] ... loat: The average of the sums of absolute differences for each shuffled permutation of the list.\nYou should write self-contained code starting with:\n```\nimport itertools\nfrom random import shuffle\ndef task_func(numbers=list(range(1, 3))):\n```"
    }
  ],
  "target": "    permutations = list(itertools.permutations(numbers))\n    sum_diffs = 0\n\n    for perm in permutations:\n        perm = list(perm)\n        shuffle(perm)\n        diffs = [abs(perm[i] - perm[i+1]) for i in range(len(perm)-1)]\n        sum_diffs += sum(diffs)\n\n    avg_sum_diffs = sum_diffs / len(permutations)\n    \n    return avg_sum_diffs",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "BigCodeBench/0",
    "entry_point": "task_func",
    "complete_prompt": "import itertools\nfrom random import shuffle\n\ndef task_func(numbers=list(range(1, 3))):\n    \"\"\"\n    Calculates the average of the sums of absolute differences between each pair of consecutive numbers \n    for all permutations of a given list.  ... [TRUNCATED 187 chars] ... age of the sums of absolute differences for each shuffled permutation of the list.\n\n    Requirements:\n    - itertools\n    - random.shuffle\n\n    Example:\n    >>> result = task_func([1, 2, 3])\n    >>> isinstance(result, float)\n    True\n    \"\"\"\n",
    "code_prompt": "import itertools\nfrom random import shuffle\ndef task_func(numbers=list(range(1, 3))):\n",
    "test": "import unittest\nfrom unittest.mock import patch\nfrom random import seed, shuffle\nimport itertools\nclass TestCases(unittest.TestCase):\n    def test_default_numbers(self):\n        # Test with default number range (1 to 10) to check that the res ... [TRUNCATED 2578 chars] ...  x: seed(1) or shuffle(x)):\n            result1 = task_func([1, 2, 3])\n        with patch('random.shuffle', side_effect=lambda x: seed(1) or shuffle(x)):\n            result2 = task_func([1, 2, 4])\n        self.assertNotEqual(result1, result2)"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{prompt}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | `str` | `instruct` | Evaluation mode: "complete" (docstring completion) or "instruct" (NL instruction). Choices: ['complete', 'instruct'] |
| `version` | `str` | `default` | Dataset version. Use "default" for the latest available version. |
| `calibrate` | `bool` | `True` | Whether to prepend code_prompt to the solution for function signature alignment. |

## Sandbox Configuration

This benchmark requires a sandbox environment for code execution.

```json
{
  "image": "bigcodebench/bigcodebench-evaluate:latest",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  },
  "memory_limit": "4g"
}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bigcodebench \
    --sandbox '{"enabled": true}' \
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
    datasets=['bigcodebench'],
    sandbox={'enabled': True},
    dataset_args={
        'bigcodebench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


