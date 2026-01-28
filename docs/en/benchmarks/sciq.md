# SciQ

## Overview

SciQ is a crowdsourced science exam question dataset covering Physics, Chemistry, Biology, and other scientific domains. Most questions include supporting evidence paragraphs.

## Task Description

- **Task Type**: Science Question Answering (Multiple-Choice)
- **Input**: Science question with 4 answer choices
- **Output**: Correct answer letter (A, B, C, or D)
- **Domains**: Physics, Chemistry, Biology, Earth Science, etc.

## Key Features

- Crowdsourced science exam questions
- Multiple scientific domains covered
- Supporting evidence paragraphs available
- Tests scientific knowledge and reasoning
- Suitable for science comprehension evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on test split
- Simple accuracy metric

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `sciq` |
| **Dataset ID** | [extraordinarylab/sciq](https://modelscope.cn/datasets/extraordinarylab/sciq/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `ReadingComprehension` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 322.68 chars |
| Prompt Length (Min/Max) | 244 / 505 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "a30097af",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nCompounds that are capable of accepting electrons, such as o 2 or f2, are called what?\n\nA) antioxidants\nB) Oxygen\nC) residues\nD) oxidants"
    }
  ],
  "choices": [
    "antioxidants",
    "Oxygen",
    "residues",
    "oxidants"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {}
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
    --datasets sciq \
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
    datasets=['sciq'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


