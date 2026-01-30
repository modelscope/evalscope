# SIQA

## Overview

SIQA (Social Interaction QA) is a benchmark for evaluating social commonsense intelligence - understanding people's actions and their social implications. Unlike benchmarks focusing on physical knowledge, SIQA tests reasoning about human behavior.

## Task Description

- **Task Type**: Social Commonsense Reasoning
- **Input**: Context about a social situation with question and 3 answer choices
- **Output**: Most socially appropriate answer (A, B, or C)
- **Focus**: Human behavior, motivations, and social implications

## Key Features

- Tests social intelligence and emotional understanding
- Questions about people's actions and their consequences
- Covers motivations, reactions, and social norms
- 33K+ crowdsourced QA pairs
- Requires reasoning about human psychology

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on validation split
- Simple accuracy metric

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `siqa` |
| **Dataset ID** | [extraordinarylab/siqa](https://modelscope.cn/datasets/extraordinarylab/siqa/summary) |
| **Paper** | N/A |
| **Tags** | `Commonsense`, `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,954 |
| Prompt Length (Mean) | 289.05 chars |
| Prompt Length (Min/Max) | 242 / 509 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "8d09aab2",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C.\n\nWhat does Tracy need to do before this?\n\nA) make a new plan\nB) Go home and see Riley\nC) Find somewhere to go"
    }
  ],
  "choices": [
    "make a new plan",
    "Go home and see Riley",
    "Find somewhere to go"
  ],
  "target": "C",
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
    --datasets siqa \
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
    datasets=['siqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


