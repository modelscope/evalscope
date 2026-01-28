# MuSR


## Overview

MuSR (Multistep Soft Reasoning) is a benchmark for evaluating complex reasoning abilities through narrative-based problems. It includes murder mysteries, object placements, and team allocation scenarios requiring multi-step inference.

## Task Description

- **Task Type**: Complex Reasoning (Multiple-Choice)
- **Input**: Narrative scenario with question and answer choices
- **Output**: Correct answer letter (A-F)
- **Domains**: Murder mysteries, object tracking, team allocation

## Key Features

- Narrative-based reasoning problems
- Requires multi-step logical inference
- Three distinct reasoning domains
- Tests constraint satisfaction and deduction
- Longer context requiring careful reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting
- Three subsets: `murder_mysteries`, `object_placements`, `team_allocation`
- Simple accuracy metric
- Challenging benchmark requiring careful reading


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `musr` |
| **Dataset ID** | [AI-ModelScope/MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 756 |
| Prompt Length (Mean) | 4891.57 chars |
| Prompt Length (Min/Max) | 2812 / 7537 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `murder_mysteries` | 250 | 5743.1 | 4056 | 7537 |
| `object_placements` | 256 | 5294.0 | 3735 | 7525 |
| `team_allocation` | 250 | 3627.93 | 2812 | 4351 |

## Sample Example

**Subset**: `murder_mysteries`

```json
{
  "input": [
    {
      "id": "5ec1a7bd",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B. Think step by step before answering.\n\nIn an adrenaline inducing ... [TRUNCATED] ... and wronged, over and over, at the same sight. It was quite a sight. \n\nWinston, shuffling back to the station, was left with one thought - Looks like Mackenzie had quite an eventful week.\n\nWho is the most likely murderer?\n\nA) Mackenzie\nB) Ana"
    }
  ],
  "choices": [
    "Mackenzie",
    "Ana"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets musr \
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
    datasets=['musr'],
    dataset_args={
        'musr': {
            # subset_list: ['murder_mysteries', 'object_placements', 'team_allocation']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


