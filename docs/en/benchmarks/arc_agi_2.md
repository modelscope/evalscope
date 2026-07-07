# ARC-AGI-2


## Overview

ARC-AGI-2 (Abstraction and Reasoning Corpus for Artificial General Intelligence 2) is a benchmark designed to measure an AI system's ability to efficiently acquire new skills on-the-fly, using only a handful of demonstrations. It evaluates abstract reasoning and pattern recognition through grid transformation tasks.

## Task Description

- **Task Type**: Abstract Reasoning / Pattern Recognition
- **Input**: A series of input-output grid pairs (demonstrations) followed by a test input grid
- **Output**: The predicted output grid matching the inferred transformation rule
- **Grid Format**: 2D arrays of integers (0-9), variable sizes (up to 30x30)

## Key Features

- 1,000 public training tasks and 120 public evaluation tasks
- Each task provides 2-10 demonstration input/output pairs
- Models must infer the transformation rule from demonstrations
- Tests abstract reasoning without reliance on learned knowledge
- Pixel-perfect output required (exact grid match)

## Evaluation Notes

- Scoring is based on **exact grid match** (shape and all values must be identical)
- Models must output the grid as a JSON 2D array
- Zero-shot evaluation (demonstrations are provided within each task)
- Designed to be solvable by humans but challenging for AI


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `arc_agi_2` |
| **Dataset ID** | [evalscope/arc-agi-2](https://modelscope.cn/datasets/evalscope/arc-agi-2/summary) |
| **Paper** | N/A |
| **Tags** | `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_hat_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 120 |
| Prompt Length (Mean) | 8026.79 chars |
| Prompt Length (Min/Max) | 2437 / 25471 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "7b97143f",
      "content": "You are an expert at abstract reasoning and pattern recognition. Given input-output grid pairs as examples, you must figure out the transformation rule and apply it to a new test input to produce the correct output grid."
    },
    {
      "id": "bdaa2b22",
      "content": "You are given a series of input-output grid pairs as examples. Each grid is a 2D array of integers (0-9). Study the pattern in the examples, then predict the output for the test input.\n\nExamples:\nExample 1:\nInput: [[0, 0, 0, 0, 0, 0, 0, 0, 0, ... [TRUNCATED 7482 chars] ... 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]\n\nProvide the output grid as a JSON 2D array. Only output the JSON array, nothing else."
    }
  ],
  "target": "[[8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8], [8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [8, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 0], [ ... [TRUNCATED 1596 chars] ... ], [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 8, 0], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0]]",
  "id": 0,
  "group_id": 0
}
```

## Prompt Template

**System Prompt:**
```text
You are an expert at abstract reasoning and pattern recognition. Given input-output grid pairs as examples, you must figure out the transformation rule and apply it to a new test input to produce the correct output grid.
```

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets arc_agi_2 \
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
    datasets=['arc_agi_2'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


