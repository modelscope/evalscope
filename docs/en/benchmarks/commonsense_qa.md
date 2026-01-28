# CommonsenseQA

## Overview

CommonsenseQA is a benchmark for evaluating AI models' ability to answer questions that require commonsense reasoning about the world. Questions are designed to require background knowledge not explicitly stated in the question.

## Task Description

- **Task Type**: Multiple-Choice Commonsense Reasoning
- **Input**: Question requiring commonsense knowledge with 5 choices
- **Output**: Correct answer letter (A-E)
- **Focus**: World knowledge and commonsense inference

## Key Features

- Questions generated from ConceptNet knowledge graph
- Requires different types of commonsense knowledge
- 5 answer choices per question
- Tests reasoning about everyday concepts and relationships
- Human-validated questions

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on validation split
- Simple accuracy metric

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `commonsense_qa` |
| **Dataset ID** | [extraordinarylab/commonsense-qa](https://modelscope.cn/datasets/extraordinarylab/commonsense-qa/summary) |
| **Paper** | N/A |
| **Tags** | `Commonsense`, `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,221 |
| Prompt Length (Mean) | 326.11 chars |
| Prompt Length (Min/Max) | 257 / 537 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "c9b96cfc",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E.\n\nA revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\n\nA) bank\nB) library\nC) department store\nD) mall\nE) new york"
    }
  ],
  "choices": [
    "bank",
    "library",
    "department store",
    "mall",
    "new york"
  ],
  "target": "A",
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
    --datasets commonsense_qa \
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
    datasets=['commonsense_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


