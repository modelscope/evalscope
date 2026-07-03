# CMATH

## Overview

CMATH is a Chinese elementary school mathematics benchmark containing 1,698 problems across grades 1-6. It evaluates the mathematical reasoning capabilities of language models on Chinese-language math word problems at increasing difficulty levels.

## Task Description

- **Task Type**: Chinese Mathematical Word Problem Solving
- **Input**: Chinese math word problem (elementary school level)
- **Output**: Step-by-step reasoning with numerical answer
- **Difficulty**: Grade 1 (easiest) to Grade 6 (hardest)

## Key Features

- 1,098 test problems + 600 validation problems
- Six grade levels (1-6) for fine-grained difficulty analysis
- Problems are in Chinese, testing language-specific reasoning
- Simple numerical answers (integers or decimals)
- Metadata includes reasoning steps count and digit complexity

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Numeric accuracy metric for answer comparison
- Results can be broken down by grade level
- Chinese prompt template used by default

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cmath` |
| **Dataset ID** | [evalscope/cmath](https://modelscope.cn/datasets/evalscope/cmath/summary) |
| **Paper** | N/A |
| **Tags** | `Chinese`, `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,098 |
| Prompt Length (Mean) | 70.14 chars |
| Prompt Length (Min/Max) | 38 / 191 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Grade 1` | 164 | 60.65 | 45 | 99 |
| `Grade 2` | 253 | 63.3 | 38 | 168 |
| `Grade 3` | 237 | 69.49 | 49 | 148 |
| `Grade 4` | 120 | 75.33 | 51 | 114 |
| `Grade 5` | 126 | 76.79 | 49 | 124 |
| `Grade 6` | 198 | 80.16 | 42 | 191 |

## Sample Example

**Subset**: `Grade 1`

```json
{
  "input": [
    {
      "id": "6e98e414",
      "content": "妈咪买了3盒茶叶，一盒茶叶有6小包，一共买了多少小包茶叶？\n请一步一步推理，最后将答案放在\\boxed{}中。\n"
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "subset_key": "Grade 1",
  "metadata": {
    "reasoning_step": 1,
    "num_digits": 2
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
请一步一步推理，最后将答案放在\boxed{{}}中。

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cmath \
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
    datasets=['cmath'],
    dataset_args={
        'cmath': {
            # subset_list: ['Grade 1', 'Grade 2', 'Grade 3']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


