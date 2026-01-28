# MBPP


## Overview

MBPP (Mostly Basic Python Problems) is a benchmark consisting of approximately 1,000 crowd-sourced Python programming problems designed for entry-level programmers. It evaluates a model's ability to understand problem descriptions and generate correct Python code.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Natural language task description with test cases
- **Output**: Python function implementation
- **Difficulty**: Entry-level programming problems

## Key Features

- ~1,000 crowd-sourced programming problems
- Covers programming fundamentals and standard library usage
- Each problem includes task description, solution, and 3 test cases
- Problems are designed to be solvable by entry-level programmers
- Automatic evaluation through test case execution

## Evaluation Notes

- Default configuration uses **3-shot** examples
- **Security Warning**: Sandbox environment is required for safe code execution. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Supports `pass@k` metric calculation
- Default timeout is 20 seconds per problem
- Code is extracted from `[BEGIN]...[DONE]` blocks if present


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mbpp` |
| **Dataset ID** | [google-research-datasets/mbpp](https://modelscope.cn/datasets/google-research-datasets/mbpp/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 3-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `prompt` |
| **Aggregation** | `mean_and_pass_at_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 500 |
| Prompt Length (Mean) | 1872.34 chars |
| Prompt Length (Min/Max) | 1727 / 5896 chars |

## Sample Example

**Subset**: `full`

```json
{
  "input": [
    {
      "id": "f9c5e33a",
      "content": "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert ... [TRUNCATED] ... unction to remove first and last occurrence of a given character from the string. Your code should pass these tests:\n\nassert remove_Occ(\"hello\",\"l\") == \"heo\"\nassert remove_Occ(\"abcda\",\"a\") == \"bcd\"\nassert remove_Occ(\"PHP\",\"P\") == \"H\"\n[BEGIN]\n"
    }
  ],
  "target": "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "test_list": [
      "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
      "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
      "assert remove_Occ(\"PHP\",\"P\") == \"H\""
    ],
    "task_id": 11,
    "test_setup_code": ""
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

<details>
<summary>Few-shot Template</summary>

```text
You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:

assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
[BEGIN]
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)
[DONE]
You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
[BEGIN]
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
[DONE]
You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:

assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
[BEGIN]
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums
[DONE]
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
[BEGIN]

```

</details>

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
    --datasets mbpp \
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
    datasets=['mbpp'],
    use_sandbox=True,
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


