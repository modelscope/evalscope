# QASC

## Overview

QASC (Question Answering via Sentence Composition) is a question-answering dataset with a focus on multi-hop sentence composition. It consists of 9,980 8-way multiple-choice questions about grade school science, requiring models to combine multiple facts to arrive at the correct answer.

## Task Description

- **Task Type**: Multi-hop Science Question Answering (Multiple-Choice)
- **Input**: Science question with 8 answer choices
- **Output**: Correct answer letter
- **Focus**: Sentence composition and multi-hop reasoning

## Key Features

- 9,980 grade school science questions
- 8-way multiple-choice format
- Requires composing two facts to answer
- Tests multi-hop reasoning over scientific knowledge
- Annotated with supporting facts for each question

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on validation split
- Simple accuracy metric
- Useful for evaluating compositional reasoning

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `qasc` |
| **Dataset ID** | [extraordinarylab/qasc](https://modelscope.cn/datasets/extraordinarylab/qasc/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 926 |
| Prompt Length (Mean) | 362.58 chars |
| Prompt Length (Min/Max) | 283 / 505 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d92a545e",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H.\n\nClimate is generally described in terms of what?\n\nA) sand\nB) occurs over a wide range\nC) forests\nD) Global warming\nE) rapid changes occur\nF) local weather conditions\nG) measure of motion\nH) city life"
    }
  ],
  "choices": [
    "sand",
    "occurs over a wide range",
    "forests",
    "Global warming",
    "rapid changes occur",
    "local weather conditions",
    "measure of motion",
    "city life"
  ],
  "target": "F",
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
    --datasets qasc \
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
    datasets=['qasc'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


