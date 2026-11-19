# Qwen-Image-Bench


## Overview

Qwen-Image-Bench measures how well text-to-image models handle creative tasks.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: A Chinese or English prompt
- **Output**: A generated image
- **Domain**: Image quality, aesthetics, prompt alignment, real-world fidelity, and creativity

## Key Features

- 1,000 bilingual prompts. Set `extra_params.language` to `cn` (default) or `en`.
- Score existing images with a local JSONL (`ID`, `prompt`, `dims_en`, and absolute `image_path`).
  Set `model_task='image_generation'` and omit `model`.

## Evaluation Notes

- Configure `judge.models` with an OpenAI-compatible endpoint serving the official Q-Judger
  (`Qwen/Qwen-Image-Bench`). See the [official inference settings](https://github.com/QwenLM/Qwen-Image-Bench#inference-parameters).
- Scores range from 0 to 100. Higher is better. The report shows overall and breakdown scores with their valid image counts.


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
      "id": "58f90ec0",
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
