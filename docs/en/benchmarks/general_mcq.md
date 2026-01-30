# General-MCQ


## Overview

General-MCQ is a customizable multiple-choice question answering benchmark for evaluating language models. It supports flexible data formats and variable number of answer choices.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Question with 2-10 answer choices (A through J)
- **Output**: Selected answer choice
- **Flexibility**: Supports custom datasets via local files

## Key Features

- Flexible number of choices (A through J)
- Custom dataset support via local file loading
- Chinese single-answer prompt template
- Configurable few-shot examples
- Accuracy-based evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- Train split: **dev**, Eval split: **val**
- See [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#mcq) for dataset format


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `general_mcq` |
| **Dataset ID** | `general_mcq` |
| **Paper** | N/A |
| **Tags** | `Custom`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `val` |
| **Train Split** | `dev` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_mcq \
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
    datasets=['general_mcq'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


