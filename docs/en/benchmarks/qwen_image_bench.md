# Qwen-Image-Bench


## Overview

Qwen-Image-Bench evaluates text-to-image models on professional creative tasks using the official Q-Judger.

## Task Description

- **Task Type**: Text-to-image generation and evaluation
- **Input**: A Chinese or English creative prompt
- **Output**: An image evaluated against applicable rubrics
- **Domain**: Image quality, aesthetics, alignment, real-world fidelity, and creative generation

## Key Features

- 1,000 bilingual prompts with 56 facets grouped into five dimensions.
- Supports both image generation and evaluation of existing images via a local JSONL manifest.

## Evaluation Notes

- Set `judge.models` to an OpenAI-compatible vision endpoint serving the official `Qwen/Qwen-Image-Bench`
  weights. Serve with thinking enabled and reasoning separated from the final JSON. Use temperature 0,
  top-k 1, top-p 1, repetition penalty 1.05, seed 42, and a 4,096-token generation limit for official settings.
- Set `extra_params.language` to `cn` (default) or `en`. Each run reports scores for the selected language.
- For existing images, use a local JSONL with `ID`, `dims_en`, `prompt` (or the selected official
  `prompt_cn`/`prompt_en` field), and an absolute `image_path`. Set `model_task='image_generation'`
  and leave `model` unset to score these images.
- Grades 0, 1, and 2 map to 0, 60, and 100. Inapplicable facets and failed requests are excluded from the
  corresponding averages. Each image is averaged across its scored dimensions before averaging images.
- The overall score, dimensions, and subdimensions report their respective valid image counts.

This configuration scores existing images against the English prompts.

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model_id='my-image-model',
    model_task='image_generation',
    datasets=['qwen_image_bench'],
    dataset_args={'qwen_image_bench': {
        'dataset_id': '/absolute/path/to/manifest.jsonl',
        'extra_params': {'language': 'en'},
    }},
    judge={'models': [{
        'model_id': 'Qwen/Qwen-Image-Bench',
        'api_url': 'http://127.0.0.1:8000/v1',
        'generation_config': {
            'temperature': 0,
            'top_p': 1,
            'seed': 42,
            'max_tokens': 4096,
            'extra_body': {
                'top_k': 1,
                'repetition_penalty': 1.05,
                'chat_template_kwargs': {'enable_thinking': True},
            },
        },
    }]},
))
```


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `qwen_image_bench` |
| **Dataset ID** | [Qwen/Qwen-Image-Bench](https://modelscope.cn/datasets/Qwen/Qwen-Image-Bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2605.28091) |
| **Tags** | `TextToImage` |
| **Metrics** | `qwen_image_bench_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 83.05 chars |
| Prompt Length (Min/Max) | 25 / 317 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "b8c6bf0d",
      "content": "一只拆开的机械手表放在黑色工作台上，制表师用镊子夹起齿轮，特写镊子夹起齿轮，银色与暗灰色调，零部件结构精密，金属边缘细节清晰。"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": 1,
    "prompt": "一只拆开的机械手表放在黑色工作台上，制表师用镊子夹起齿轮，特写镊子夹起齿轮，银色与暗灰色调，零部件结构精密，金属边缘细节清晰。",
    "dimensions": [
      "Quality",
      "Aesthetics",
      "Alignment"
    ],
    "image_path": "",
    "category": "",
    "tags": []
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `cn` | Language of the generation prompts (cn or en). Choices: ['cn', 'en'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets qwen_image_bench \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['qwen_image_bench'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'qwen_image_bench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
