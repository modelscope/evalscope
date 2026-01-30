# GenAI-Bench


## Overview

GenAI-Bench is a comprehensive text-to-image benchmark featuring 1600 prompts designed to evaluate image generation models across diverse categories and complexity levels.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompts with varying complexity (basic and advanced)
- **Output**: Generated images evaluated using VQAScore
- **Size**: 1600 prompts

## Key Features

- Large-scale prompt collection for thorough evaluation
- Categorized prompts (basic vs advanced)
- Uses VQAScore for semantic alignment assessment
- Rich metadata including category tags
- Supports both generation and pre-existing image evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **VQAScore** for semantic alignment
- Evaluates images from the **test** split
- Prompts categorized as 'basic' or 'advanced' based on complexity
- Part of the T2V-Eval-Prompts dataset collection


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `genai_bench` |
| **Dataset ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **Paper** | N/A |
| **Tags** | `TextToImage` |
| **Metrics** | `VQAScore` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,600 |
| Prompt Length (Mean) | 67.42 chars |
| Prompt Length (Min/Max) | 14 / 192 chars |

## Sample Example

**Subset**: `GenAI-Bench-1600`

```json
{
  "input": [
    {
      "id": "24dc1965",
      "content": "A baker pulling freshly baked bread out of an oven in a bakery."
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "GenAI-Bench1600_0",
    "prompt": "A baker pulling freshly baked bread out of an oven in a bakery.",
    "category": "basic",
    "tags": {
      "advanced": [],
      "basic": [
        "Attribute",
        "Scene",
        "Spatial Relation",
        "Action Relation"
      ]
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
    --datasets genai_bench \
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
    datasets=['genai_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


