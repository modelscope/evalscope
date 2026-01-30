# MultiPL-E MBPP


## Overview

MultiPL-E MBPP is a multilingual code generation benchmark derived from MBPP (Mostly Basic Python Programming). It extends the original MBPP to 18 programming languages, enabling cross-lingual evaluation of code generation capabilities.

## Task Description

- **Task Type**: Multilingual Code Generation
- **Input**: Programming problem prompt with docstring
- **Output**: Complete code solution that passes test cases
- **Languages**: 18 languages (C++, TypeScript, Shell, C#, Go, Java, Lua, JavaScript, PHP, Perl, Racket, R, Rust, Scala, Swift, Ruby, D, Julia)

## Key Features

- Multilingual evaluation across 18 programming languages
- Execution-based evaluation with test cases
- Supports pass@k metric for code generation
- Docker sandbox environment for safe code execution
- Derived from MBPP with consistent problem difficulty

## Evaluation Notes

- **Sandbox Required**: Requires sandbox environment for safe code execution
- Default evaluation uses **test** split
- Primary metric: **Accuracy** with **pass@k** aggregation
- Timeout: 30 seconds per test case
- See [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for setup


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `multiple_mbpp` |
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
| Total Samples | 6,987 |
| Prompt Length (Mean) | 379.32 chars |
| Prompt Length (Min/Max) | 266 / 1002 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `mbpp-cpp` | 397 | 407.55 | 317 | 1002 |
| `mbpp-ts` | 390 | 359.57 | 295 | 674 |
| `mbpp-sh` | 382 | 362.53 | 296 | 696 |
| `mbpp-cs` | 386 | 549.09 | 481 | 867 |
| `mbpp-go` | 374 | 402.21 | 332 | 723 |
| `mbpp-java` | 386 | 553 | 474 | 889 |
| `mbpp-lua` | 397 | 332.52 | 277 | 651 |
| `mbpp-js` | 397 | 325.48 | 270 | 645 |
| `mbpp-php` | 397 | 336.17 | 280 | 655 |
| `mbpp-pl` | 396 | 337.87 | 282 | 657 |
| `mbpp-rkt` | 397 | 342.25 | 287 | 660 |
| `mbpp-r` | 397 | 326.42 | 271 | 644 |
| `mbpp-rs` | 354 | 351.9 | 285 | 669 |
| `mbpp-scala` | 396 | 434.01 | 364 | 750 |
| `mbpp-swift` | 396 | 352.64 | 285 | 667 |
| `mbpp-rb` | 397 | 321.34 | 266 | 641 |
| `mbpp-d` | 358 | 376.49 | 308 | 693 |
| `mbpp-jl` | 390 | 363.01 | 292 | 686 |

## Sample Example

**Subset**: `mbpp-cpp`

```json
{
  "input": [
    {
      "id": "209e8d38",
      "content": "```cpp\n#include<assert.h>\n#include<bits/stdc++.h>\n// Write a cppthon function to identify non-prime numbers.\nbool is_not_prime(long n) {\n\n```\n\nPlease complete the above code according to the requirements in the docstring. Write the complete code and wrap it in markdown fenced code. The code should not contain `Main` function."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tests": "}\nint main() {\n    auto candidate = is_not_prime;\n    assert(candidate((2)) == (false));\n    assert(candidate((10)) == (true));\n    assert(candidate((35)) == (true));\n    assert(candidate((37)) == (false));\n}\n",
    "stop_tokens": [
      "\n}"
    ],
    "task_id": "mbpp_3_is_not_prime",
    "language": "cpp",
    "doctests": "transform"
  }
}
```

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
    --datasets multiple_mbpp \
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
    datasets=['multiple_mbpp'],
    use_sandbox=True,
    dataset_args={
        'multiple_mbpp': {
            # subset_list: ['mbpp-cpp', 'mbpp-ts', 'mbpp-sh']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


