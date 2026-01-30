# TIFA-160


## Overview

TIFA-160 is a text-to-image benchmark with 160 carefully curated prompts designed to evaluate the faithfulness and quality of generated images using automated VQA-based evaluation.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompt for image generation
- **Output**: Generated image evaluated using PickScore metric
- **Size**: 160 prompts

## Key Features

- Compact, high-quality prompt set for efficient evaluation
- Uses PickScore for human preference alignment
- Tests diverse image generation capabilities
- Supports both new generation and pre-existing image evaluation
- Reproducible evaluation pipeline

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **PickScore** for human preference alignment
- Evaluates images from the **test** split
- Part of the T2V-Eval-Prompts dataset collection


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tifa160` |
| **Dataset ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **Paper** | N/A |
| **Tags** | `TextToImage` |
| **Metrics** | `PickScore` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 160 |
| Prompt Length (Mean) | 56.13 chars |
| Prompt Length (Min/Max) | 13 / 182 chars |

## Sample Example

**Subset**: `TIFA-160`

```json
{
  "input": [
    {
      "id": "9de3e3b1",
      "content": "A Christmas tree with lights and teddy bear"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "prompt": "A Christmas tree with lights and teddy bear",
    "category": "",
    "tags": {},
    "id": "TIFA160_0",
    "image_path": ""
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tifa160 \
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
    datasets=['tifa160'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


