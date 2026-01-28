# HPD-v2


## Overview

HPD-v2 (Human Preference Dataset v2) is a text-to-image benchmark that evaluates generated images based on human preferences. It uses the HPSv2.1 score metric trained on large-scale human preference data.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompt for image generation
- **Output**: Generated image evaluated against human preferences
- **Metric**: HPSv2.1 Score

## Key Features

- Human preference-aligned evaluation metric
- Trained on large-scale human preference data
- Tests aesthetic quality and prompt alignment
- Supports diverse prompt categories
- Objective, reproducible scoring

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- HPSv2.1 Score metric measures human preference alignment
- Supports local prompt files
- Category tags available in metadata
- Can evaluate existing images or generate new ones


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hpdv2` |
| **Dataset ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **Paper** | N/A |
| **Tags** | `TextToImage` |
| **Metrics** | `HPSv2.1Score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,200 |
| Prompt Length (Mean) | 81.71 chars |
| Prompt Length (Min/Max) | 9 / 404 chars |

## Sample Example

**Subset**: `HPDv2`

```json
{
  "input": [
    {
      "id": "c971b3c7",
      "content": "Spongebob depicted in the style of Dragon Ball Z."
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "HPDv2_0",
    "prompt": "Spongebob depicted in the style of Dragon Ball Z.",
    "category": "Animation",
    "tags": {
      "category": "Animation"
    },
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
    --datasets hpdv2 \
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
    datasets=['hpdv2'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


