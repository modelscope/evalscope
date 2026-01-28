# General-VMCQ


## Overview

General-VMCQ is a customizable visual multiple-choice question answering benchmark for multimodal models. It uses MMMU-style format with image placeholders in text, supporting flexible image inputs.

## Task Description

- **Task Type**: Visual Multiple-Choice Question Answering
- **Input**: Question with `<image N>` placeholders + choice options + images
- **Output**: Selected answer choice
- **Flexibility**: Supports custom datasets via local files

## Key Features

- MMMU-style format (not OpenAI message format)
- Supports up to 100 images per sample
- Flexible image input (path, URL, or base64 data URL)
- Chain-of-thought prompt template option
- Custom dataset support via local file loading

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- Train split: **dev**, Eval split: **val**
- Images are plain strings (do not wrap in `{{"url": ...}}`)
- See [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html) for dataset format


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `general_vmcq` |
| **Dataset ID** | `general_vmcq` |
| **Paper** | N/A |
| **Tags** | `Custom`, `MCQ`, `MultiModal` |
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
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets general_vmcq \
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
    datasets=['general_vmcq'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


