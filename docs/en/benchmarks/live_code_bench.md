# Live-Code-Bench


## Overview

LiveCodeBench is a contamination-free benchmark for evaluating code generation models on real-world competitive programming problems. It continuously collects new problems from coding platforms to ensure models haven't seen the test data during training.

## Task Description

- **Task Type**: Competitive Programming / Code Generation
- **Input**: Programming problem description with input/output format
- **Output**: Complete solution code
- **Source**: Problems from LeetCode, Codeforces, and AtCoder

## Key Features

- Continuously updated with new problems post model training cutoff
- Problems from major competitive programming platforms
- Multiple test cases per problem for thorough evaluation
- Date-based filtering to control for data contamination
- Supports both local and sandbox code execution

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- **Security Warning**: By default, code is executed in the local environment. We recommend using sandbox execution. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Use `start_date` and `end_date` parameters to filter problems by date
- Default timeout is 6 seconds per test case
- Supports `pass@k` metric calculation


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `live_code_bench` |
| **Dataset ID** | [AI-ModelScope/code_generation_lite](https://modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_at_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,055 |
| Prompt Length (Mean) | 1700.76 chars |
| Prompt Length (Min/Max) | 717 / 4234 chars |

## Sample Example

**Subset**: `release_latest`

```json
{
  "input": [
    {
      "id": "6b6a6121",
      "content": "### Question:\nThere are three cards with letters $\\texttt{a}$, $\\texttt{b}$, $\\texttt{c}$ placed in a row in some order. You can do the following operation at most once: \n\n \n-  Pick two cards, and swap them.  Is it possible that the row becom ... [TRUNCATED] ... s from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n```python\n# YOUR CODE HERE\n```\n\n ### Answer: (use the provided format with backticks)\n\n"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "evaluation_sample": "{\"inputs\": [\"6\\nabc\\nacb\\nbac\\nbca\\ncab\\ncba\\n\", \"1\\nabc\\n\", \"3\\nabc\\nabc\\nabc\\n\", \"5\\ncab\\nacb\\ncba\\nbac\\nbca\\n\", \"6\\nabc\\nabc\\nabc\\nabc\\nabc\\nabc\\n\"], \"outputs\": [\"YES\\nYES\\nYES\\nNO\\nNO\\nYES\\n\", \"YES\\n\", \"YES\\nYES\\nYES\\n\", \"NO\\nYES\\nYES\\nYES\\nNO\\n\", \"YES\\nYES\\nYES\\nYES\\nYES\\nYES\\n\"], \"fn_name\": null}",
    "contest_date": "2023-08-21T00:00:00"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
### Question:
{question_content}

{format_prompt} ### Answer: (use the provided format with backticks)


```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | `str | null` | `None` | Filter problems starting from this date (YYYY-MM-DD). Null keeps all. |
| `end_date` | `str | null` | `None` | Filter problems up to this date (YYYY-MM-DD). Null keeps all. |
| `debug` | `bool` | `False` | Enable verbose debug logging and bypass certain safety checks. |

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
    --datasets live_code_bench \
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
    datasets=['live_code_bench'],
    use_sandbox=True,
    dataset_args={
        'live_code_bench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


