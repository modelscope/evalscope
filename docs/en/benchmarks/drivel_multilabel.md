# DrivelologyMultilabelClassification

## Overview

Drivelology Multi-label Classification evaluates models' ability to categorize "drivelology" text into rhetorical technique categories: inversion, wordplay, switchbait, paradox, and misdirection. Each text may belong to multiple categories.

## Task Description

- **Task Type**: Multi-label Text Classification
- **Input**: Drivelology text sample
- **Output**: One or more technique categories
- **Domain**: Linguistic analysis, rhetorical technique detection

## Key Features

- Five rhetorical technique categories
- Multi-label classification (multiple categories per sample)
- Tests understanding of linguistic creativity mechanisms
- Requires recognition of humor and irony techniques
- Detailed category definitions provided

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: F1 (weighted, micro, macro), Exact Match
- Aggregation method: F1 weighted
- Categories: inversion, wordplay, switchbait, paradox, misdirection

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `drivel_multilabel` |
| **Dataset ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ` |
| **Metrics** | `f1_weighted`, `f1_micro`, `f1_macro`, `exact_match` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `f1_weighted` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 600 |
| Prompt Length (Mean) | 1637.18 chars |
| Prompt Length (Min/Max) | 1580 / 2041 chars |

## Sample Example

**Subset**: `multi-label-classification`

```json
{
  "input": [
    {
      "id": "0e8acb03",
      "content": [
        {
          "text": "#Instruction#:\nClassify the given text into one or more of the following categories: inversion, wordplay, switchbait, paradox, and misdirection.\n\n#Definitions#:\n- inversion: This technique takes a well-known phrase, cliché, or social script a ... [TRUNCATED] ... e should be of the following format: 'ANSWER: [LETTERS]' (without quotes) where [LETTER]S is one or more of A,B,C,D,E.\n\nText to classify: 後天的努力比什麼都重要，所以今天和明天休息。\n\nA) A. inversion\nB) B. wordplay\nC) C. switchbait\nD) D. paradox\nE) E. misdirection"
        }
      ]
    }
  ],
  "choices": [
    "A. inversion",
    "B. wordplay",
    "C. switchbait",
    "D. paradox",
    "E. misdirection"
  ],
  "target": "AB",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "text": "後天的努力比什麼都重要，所以今天和明天休息。",
    "label": [
      "inversion",
      "wordplay"
    ],
    "target_letters": "AB"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

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
    --datasets drivel_multilabel \
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
    datasets=['drivel_multilabel'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


