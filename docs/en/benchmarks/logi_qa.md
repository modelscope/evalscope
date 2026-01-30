# LogiQA

## Overview

LogiQA is a benchmark for evaluating logical reasoning abilities, sourced from expert-written questions originally designed for testing human logical reasoning skills in standardized examinations.

## Task Description

- **Task Type**: Logical Reasoning (Multiple-Choice)
- **Input**: Context passage, question, and 4 answer choices
- **Output**: Correct answer letter (A, B, C, or D)
- **Focus**: Deductive reasoning, logical inference

## Key Features

- Expert-written logical reasoning questions
- Requires formal logical analysis
- Tests various reasoning patterns (syllogisms, conditionals, etc.)
- Questions from standardized examinations
- Challenging for both humans and models

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should follow "ANSWER: [LETTER]" format
- Evaluates on test split
- Simple accuracy metric

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `logi_qa` |
| **Dataset ID** | [extraordinarylab/logiqa](https://modelscope.cn/datasets/extraordinarylab/logiqa/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 651 |
| Prompt Length (Mean) | 1052.17 chars |
| Prompt Length (Min/Max) | 389 / 1911 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "a7ac2424",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nIn the planning of a new district in a township, it w ... [TRUNCATED] ... ) Civic Park is north of the administrative service area.\nB) The leisure area is southwest of the cultural area.\nC) The cultural district is in the northeast of the business district.\nD) The business district is southeast of the leisure area."
    }
  ],
  "choices": [
    "Civic Park is north of the administrative service area.",
    "The leisure area is southwest of the cultural area.",
    "The cultural district is in the northeast of the business district.",
    "The business district is southeast of the leisure area."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

*Note: Some content was truncated for display.*

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
    --datasets logi_qa \
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
    datasets=['logi_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


