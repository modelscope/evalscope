# SimpleQA


## Overview

SimpleQA is a benchmark by OpenAI designed to evaluate language models' ability to answer short, fact-seeking questions accurately. It focuses on measuring factual accuracy with clear grading criteria for correct, incorrect, and not-attempted answers.

## Task Description

- **Task Type**: Factual Question Answering
- **Input**: Simple factual question
- **Output**: Concise factual answer
- **Grading**: CORRECT, INCORRECT, or NOT_ATTEMPTED

## Key Features

- Short, fact-seeking questions with unambiguous answers
- Clear grading criteria for accuracy evaluation
- Distinguishes between incorrect answers and abstentions
- Uses LLM-as-judge for semantic answer comparison
- Tests factual knowledge and calibration

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM judge for answer grading (semantic matching)
- Three-way classification: is_correct, is_incorrect, is_not_attempted
- Allows hedging if correct information is included
- Tests models' ability to admit uncertainty appropriately


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `simple_qa` |
| **Dataset ID** | [evalscope/SimpleQA](https://modelscope.cn/datasets/evalscope/SimpleQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `QA` |
| **Metrics** | `is_correct`, `is_incorrect`, `is_not_attempted` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,326 |
| Prompt Length (Mean) | 118.47 chars |
| Prompt Length (Min/Max) | 48 / 403 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "9dd57f4c",
      "content": "Answer the question:\n\nWho received the IEEE Frank Rosenblatt Award in 2010?"
    }
  ],
  "target": "Michio Sugeno",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "topic": "Science and technology",
    "answer_type": "Person",
    "urls": [
      "https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
      "https://ieeexplore.ieee.org/author/37271220500",
      "https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
      "https://www.nxtbook.com/nxtbooks/ieee/awards_2010/index.php?startid=21#/p/20"
    ]
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the question:

{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets simple_qa \
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
    datasets=['simple_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


