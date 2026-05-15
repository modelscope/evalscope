# General-MCQ


## Overview

General-MCQ is a customizable multiple-choice question answering benchmark for evaluating language models. It supports flexible data formats and variable number of answer choices.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Question with 2-10 answer choices (A through J)
- **Output**: Selected answer choice(s)
- **Flexibility**: Supports custom datasets via local files, single or multiple correct answers

## Key Features

- Flexible number of choices (A through J)
- Custom dataset support via local file loading
- Chinese single-/multiple-answer prompt templates (optional CoT variants)
- Configurable few-shot examples
- Accuracy-based evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation with single-answer template
- Set `extra_params.multiple_correct=True` to evaluate questions with multiple correct answers
- Set `extra_params.use_cot=True` to switch to chain-of-thought prompt templates
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

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multiple_correct` | `bool` | `False` | Whether the dataset contains questions with multiple correct answers. When True, switches to the multiple-answer prompt template and parser, and requires the `answer` field to be a list of letters (e.g., ["A", "C"]). |
| `use_cot` | `bool` | `False` | Whether to use the chain-of-thought (CoT) prompt template variant. |

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
    dataset_args={
        'general_mcq': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


