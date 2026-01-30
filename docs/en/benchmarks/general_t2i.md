# general_t2i


## Overview

General Text-to-Image is a customizable benchmark adapter for evaluating text-to-image generation models with user-provided prompts and images.

## Task Description

- **Task Type**: Custom Text-to-Image Evaluation
- **Input**: User-provided text prompts
- **Output**: Generated images evaluated using configurable metrics
- **Flexibility**: Supports local prompt files

## Key Features

- Flexible custom evaluation framework
- Supports local prompt file loading
- Configurable metrics (default: PickScore)
- User-defined subset naming from file path
- Compatible with pre-generated images

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **PickScore** (configurable)
- Supports custom dataset paths and prompt files
- Automatically extracts subset name from file path
- Can evaluate existing images via `image_path` field


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `general_t2i` |
| **Dataset ID** | `general_t2i` |
| **Paper** | N/A |
| **Tags** | `Custom`, `TextToImage` |
| **Metrics** | `PickScore` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_t2i \
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
    datasets=['general_t2i'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


