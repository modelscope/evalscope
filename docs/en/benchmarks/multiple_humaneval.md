# MultiPL-E HumanEval


## Overview

MultiPL-E HumanEval is a multilingual code generation benchmark derived from OpenAI's HumanEval. It extends the original HumanEval to 18 programming languages, enabling cross-lingual evaluation of code generation capabilities.

## Task Description

- **Task Type**: Multilingual Code Generation
- **Input**: Programming problem prompt with function signature and docstring
- **Output**: Complete function implementation that passes test cases
- **Languages**: 18 languages (C++, TypeScript, Shell, C#, Go, Java, Lua, JavaScript, PHP, Perl, Racket, R, Rust, Scala, Swift, Ruby, D, Julia)

## Key Features

- Multilingual evaluation across 18 programming languages
- Execution-based evaluation with test cases
- Supports pass@k metric for code generation
- Docker sandbox environment for safe code execution
- Derived from HumanEval with consistent problem difficulty

## Evaluation Notes

- **Sandbox Required**: Requires sandbox environment for safe code execution
- Default evaluation uses **test** split
- Primary metric: **Accuracy** with **pass@k** aggregation
- Timeout: 30 seconds per test case
- See [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for setup


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `multiple_humaneval` |
| **Dataset ID** | [evalscope/MultiPL-E](https://modelscope.cn/datasets/evalscope/MultiPL-E/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_at_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,864 |
| Prompt Length (Mean) | 696.59 chars |
| Prompt Length (Min/Max) | 285 / 2463 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `humaneval-cpp` | 161 | 797.74 | 351 | 2057 |
| `humaneval-ts` | 159 | 639.27 | 320 | 1529 |
| `humaneval-sh` | 158 | 648.14 | 325 | 1534 |
| `humaneval-cs` | 158 | 983.67 | 538 | 2335 |
| `humaneval-go` | 154 | 701.14 | 351 | 1616 |
| `humaneval-java` | 158 | 1009.53 | 531 | 2463 |
| `humaneval-lua` | 161 | 620.31 | 294 | 1497 |
| `humaneval-js` | 161 | 611.34 | 287 | 1490 |
| `humaneval-php` | 161 | 632.01 | 298 | 1551 |
| `humaneval-pl` | 161 | 611.27 | 296 | 1481 |
| `humaneval-rkt` | 161 | 634.22 | 304 | 1537 |
| `humaneval-r` | 161 | 607.78 | 285 | 1484 |
| `humaneval-rs` | 156 | 686.67 | 313 | 1591 |
| `humaneval-scala` | 160 | 832.81 | 423 | 1998 |
| `humaneval-swift` | 158 | 660.88 | 323 | 1556 |
| `humaneval-rb` | 161 | 611.55 | 289 | 1474 |
| `humaneval-d` | 156 | 640.72 | 328 | 1501 |
| `humaneval-jl` | 159 | 616.4 | 303 | 1478 |

## Sample Example

**Subset**: `humaneval-cpp`

```json
{
  "input": [
    {
      "id": "8f096947",
      "content": "```cpp\n#include<assert.h>\n#include<bits/stdc++.h>\n// Check if in given vector of numbers, are any two numbers closer to each other than\n// given threshold.\n// >>> has_close_elements((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.0f}) ... [TRUNCATED] ... ents(std::vector<float> numbers, float threshold) {\n\n```\n\nPlease complete the above code according to the requirements in the docstring. Write the complete code and wrap it in markdown fenced code. The code should not contain `Main` function."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tests": "}\nint main() {\n    auto candidate = has_close_elements;\n    assert(candidate((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.9f, (float)4.0f, (float)5.0f, (float)2.2f})), (0.3f)) == (true));\n    assert(candidate((std::vector<float>({( ... [TRUNCATED] ... (std::vector<float>({(float)1.1f, (float)2.2f, (float)3.1f, (float)4.1f, (float)5.1f})), (1.0f)) == (true));\n    assert(candidate((std::vector<float>({(float)1.1f, (float)2.2f, (float)3.1f, (float)4.1f, (float)5.1f})), (0.5f)) == (false));\n}\n",
    "stop_tokens": [
      "\n}"
    ],
    "task_id": "HumanEval_0_has_close_elements",
    "language": "cpp",
    "doctests": "transform"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{prompt}
```

## Sandbox Configuration

This benchmark requires a sandbox environment for code execution.

```json
{
  "image": "volcengine/sandbox-fusion:server-20250609",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {},
    "multi_code_executor": {}
  },
  "memory_limit": "2g",
  "cpu_limit": "2.0"
}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets multiple_humaneval \
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
    datasets=['multiple_humaneval'],
    use_sandbox=True,
    dataset_args={
        'multiple_humaneval': {
            # subset_list: ['humaneval-cpp', 'humaneval-ts', 'humaneval-sh']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


