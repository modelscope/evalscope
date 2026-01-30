# HellaSwag


## Overview

HellaSwag is a benchmark for evaluating commonsense natural language inference, specifically testing a model's ability to complete sentences describing everyday situations. The dataset uses adversarial filtering to create challenging distractors that are grammatically correct but semantically implausible.

## Task Description

- **Task Type**: Multiple-Choice Sentence Completion
- **Input**: Context describing an activity or situation
- **Output**: Most plausible continuation from 4 choices (A, B, C, D)
- **Domain**: Everyday activities and commonsense scenarios

## Key Features

- 70,000+ questions testing grounded commonsense inference
- Contexts derived from ActivityNet and WikiHow
- Adversarially-filtered incorrect endings
- Requires understanding of typical event sequences
- Tests physical and social commonsense reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on the validation split
- Endings are preprocessed to clean formatting artifacts
- Context combines `ctx_a` and `ctx_b` fields
- Activity labels available in metadata for analysis


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hellaswag` |
| **Dataset ID** | [evalscope/hellaswag](https://modelscope.cn/datasets/evalscope/hellaswag/summary) |
| **Paper** | N/A |
| **Tags** | `Commonsense`, `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 10,042 |
| Prompt Length (Mean) | 767.6 chars |
| Prompt Length (Min/Max) | 329 / 1655 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "18df47e2",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nA man is sitting on a roof. He\n\nA) is using wrap to wrap a pair of skis.\nB) is ripping level tiles off.\nC) is holding a rubik's cube.\nD) starts pulling up roofing on a roof."
    }
  ],
  "choices": [
    "is using wrap to wrap a pair of skis.",
    "is ripping level tiles off.",
    "is holding a rubik's cube.",
    "starts pulling up roofing on a roof."
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "activity_label": "Roof shingle removal"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets hellaswag \
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
    datasets=['hellaswag'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


