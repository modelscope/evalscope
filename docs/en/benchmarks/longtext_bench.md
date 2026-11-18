# LongText-Bench


## Overview

LongText-Bench evaluates whether text-to-image models accurately render long English and Chinese text.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Prompt describing a scene with specified text
- **Output**: Image containing the requested text
- **Domain**: Long text rendering

## Key Features

- 160 English and 160 Chinese prompts from X-Omni, covering 8 scenes including posters,
  webpages and slides.
- Loads the ModelScope mirror by default; the official Hugging Face source is also supported.

## Evaluation Notes

- A vision API transcribes the image. Text Score is matched tokens divided by ground-truth
  tokens (English words or Chinese characters). A bilingual overall score requires a scored result
  for each language; otherwise the report shows available language scores.
- Defaults to qwen3.8-flash on DashScope: set DASHSCOPE_API_KEY; no local VLM weights are needed.
  The official evaluator uses Qwen2.5-VL-7B, so results from different transcription models
  are not directly comparable.
- Scoring follows the official matching rules; use repeats=4 for its repeated-image protocol.
  Text Score does not assess layout or aesthetics. API failures and truncated replies are excluded;
  valid empty transcriptions score zero.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `longtext_bench` |
| **Dataset ID** | [evalscope/LongText-Bench](https://modelscope.cn/datasets/evalscope/LongText-Bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2507.22058) |
| **Tags** | `TextToImage` |
| **Metrics** | `LongTextScore` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 320 |
| Prompt Length (Mean) | 614.67 chars |
| Prompt Length (Min/Max) | 181 / 1368 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en` | 160 | 964.03 | 631 | 1368 |
| `zh` | 160 | 265.32 | 181 | 393 |

## Sample Example

**Subset**: `en`

```json
{
  "input": [
    {
      "id": "a3412209",
      "content": "A lively and bustling urban street scene captured at daytime, featuring a colorful assortment of vibrant storefronts lining the busy sidewalk, with diverse groups of pedestrians leisurely strolling past shop windows, chatting or hurrying alon ... [TRUNCATED 551 chars] ... ed \"CENTRAL PARK - 2 BLOCKS\", guiding visitors with its distinctly readable font. All signage is brightly lit, eye-catching, and clearly contrasts against the detailed backdrop of dynamic city buildings, enhancing the lively urban atmosphere."
    }
  ],
  "target": "FRESH BREWS, EVERY MORNING, EVERY DAY Locally Roasted Since 1998 CENTRAL PARK - 2 BLOCKS",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "category": "sign",
    "length": "short",
    "prompt": "A lively and bustling urban street scene captured at daytime, featuring a colorful assortment of vibrant storefronts lining the busy sidewalk, with diverse groups of pedestrians leisurely strolling past shop windows, chatting or hurrying alon ... [TRUNCATED 551 chars] ... ed \"CENTRAL PARK - 2 BLOCKS\", guiding visitors with its distinctly readable font. All signage is brightly lit, eye-catching, and clearly contrasts against the detailed backdrop of dynamic city buildings, enhancing the lively urban atmosphere.",
    "text": [
      "FRESH BREWS, EVERY MORNING, EVERY DAY",
      "Locally Roasted Since 1998",
      "CENTRAL PARK - 2 BLOCKS"
    ],
    "text_length": 15,
    "prompt_id": 0,
    "tags": [],
    "id": "",
    "image_path": ""
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets longtext_bench \
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
    datasets=['longtext_bench'],
    dataset_args={
        'longtext_bench': {
            # subset_list: ['en', 'zh']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
