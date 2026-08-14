# BoolQ-Indic


## Overview

BoolQ-Indic is a translation of the BoolQ yes/no reading-comprehension benchmark into 10 Indic
languages plus English, for evaluating multilingual passage understanding.

## Task Description

- **Task Type**: Multilingual Yes/No Reading Comprehension
- **Input**: Passage + yes/no question in one of 11 languages
- **Output**: `Yes` or `No`
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (validation split)
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`)
- All languages ship in a single dataset config; this adapter reformats by the `language` field


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `indic_boolq` |
| **Dataset ID** | [sarvamai/boolq-indic](https://modelscope.cn/datasets/sarvamai/boolq-indic/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `MultiLingual`, `ReadingComprehension` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

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
    --datasets indic_boolq \
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
    datasets=['indic_boolq'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
