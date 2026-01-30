# EvalMuse


## Overview

EvalMuse is a text-to-image benchmark that evaluates the quality and semantic alignment of generated images using fine-grained analysis with the FGA-BLIP2Score metric.

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation
- **Input**: Text prompt for image generation
- **Output**: Generated image evaluated for quality and semantic fidelity
- **Metric**: FGA-BLIP2Score (Fine-Grained Analysis with BLIP-2)

## Key Features

- Fine-grained semantic alignment evaluation
- Uses BLIP-2 vision-language model for scoring
- Evaluates both image quality and prompt adherence
- Supports diverse prompt categories
- Objective, reproducible metrics

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Only **FGA_BLIP2Score** metric is supported
- Evaluates images from the **test** split
- Can evaluate pre-generated images or generate new ones


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `evalmuse` |
| **Dataset ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **Paper** | N/A |
| **Tags** | `TextToImage` |
| **Metrics** | `FGA_BLIP2Score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 199 |
| Prompt Length (Mean) | 61.05 chars |
| Prompt Length (Min/Max) | 8 / 347 chars |

## Sample Example

**Subset**: `EvalMuse`

```json
{
  "input": [
    {
      "id": "f8b508f4",
      "content": "wide angle photograph at night, award winning interior design apartment, night outside, dark, moody dim faint lighting, wood panel walls, cozy and calm, fabrics, textiles, pillows, lamps, colorful copper brass accents, secluded, many light sources, lamps, hardwood floors, book shelf, couch, desk, plants"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "prompt": "wide angle photograph at night, award winning interior design apartment, night outside, dark, moody dim faint lighting, wood panel walls, cozy and calm, fabrics, textiles, pillows, lamps, colorful copper brass accents, secluded, many light sources, lamps, hardwood floors, book shelf, couch, desk, plants",
    "category": "",
    "tags": [
      "photograph (activity)",
      "night (attribute)",
      "interior design apartment (location)",
      "outside (location)",
      "dark (attribute)",
      "moody dim faint lighting (attribute)",
      "wood panel walls (object)",
      "cozy and calm (attribute)",
      "fabrics (material)",
      "textiles (material)",
      "pillows (object)",
      "lamps (object)",
      "colorful (attribute)",
      "copper brass accents (material)",
      "secluded (attribute)",
      "many light sources (attribute)",
      "hardwood floors (material)",
      "book shelf (object)",
      "couch (object)",
      "desk (object)",
      "plants (object)",
      "wide angle (attribute)"
    ],
    "id": "EvalMuse_0",
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
    --datasets evalmuse \
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
    datasets=['evalmuse'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


