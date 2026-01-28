# General-QA


## Overview

General-QA is a customizable question answering benchmark for evaluating language models on open-ended text generation tasks. It supports flexible data formats and configurable evaluation metrics.

## Task Description

- **Task Type**: Open-Ended Question Answering
- **Input**: Question (with optional system prompt and conversation history)
- **Output**: Free-form text answer
- **Flexibility**: Supports custom datasets via local files

## Key Features

- Flexible input format (query/answer or messages format)
- Optional system prompt support
- BLEU and Rouge evaluation metrics
- Custom dataset support via local file loading
- Extensible for various QA use cases

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Default metrics: **BLEU**, **Rouge** (Rouge-L-R as main score)
- Evaluates on **test** split
- See [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#qa) for dataset format


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `general_qa` |
| **Dataset ID** | `general_qa` |
| **Paper** | N/A |
| **Tags** | `Custom`, `QA` |
| **Metrics** | `BLEU`, `Rouge` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
请回答问题
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_qa \
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
    datasets=['general_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


