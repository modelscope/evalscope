# LongText-Bench


## Overview

LongText-Bench (X-Omni, arXiv:2507.22058) evaluates long text rendering with 160 prompts each
for English and Chinese, across 8 text-rich scenes (signboards, labeled objects, printed
materials, webpages, slides, posters, subtitles, dialogues).

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation (text rendering)
- **Input**: Text prompt describing a text-rich scene
- **Output**: Generated image transcribed by a VLM and matched against the ground-truth text
- **Domain**: Long text rendering for text-to-image models

## Key Features

- 160 English + 160 Chinese prompts across 8 text-rich scene categories from the official
  X-Omni release
- Text Score compares the VLM transcription with the ground-truth fragments, word-level for
  English and character-level for Chinese, corpus-aggregated as in the official evaluator
- Language is detected per sample from the ground-truth text, so one run covers both languages

## Evaluation Notes

- The ground truth is the dataset ``text`` field (a list of rendered fragments, joined with
  spaces like the official evaluator), passed to the scorer as the sample target
- Text Score is a Counter-intersection match ratio: word-level for English, character-level
  for Chinese (whitespace removed). Each language score is total matched / total ground
  truth tokens; the overall score averages the English and Chinese scores, as in the
  official leaderboard. Evaluating one language returns that language's score.
- Loads the ModelScope mirror by default, with 160 test examples in each of the en and zh
  configurations. Original data: [X-Omni/LongText-Bench](https://huggingface.co/datasets/X-Omni/LongText-Bench).
- Transcription uses an OpenAI-compatible vision API, defaulting to qwen3.8-flash on
  DashScope with thinking disabled. Set DASHSCOPE_API_KEY; no local VLM weights or accelerator
  are required. Metric args model, api_url, api_key and generation_config configure a different
  transcription service independently of the image generation model.
- Matching and language aggregation follow the official evaluator. The official transcription
  model is Qwen2.5-VL-7B; using qwen3.8-flash or another VLM can change the score, so those
  results must be identified by transcription model and are not directly comparable to the official results.
- Transcription failures and API token-limit truncation exclude the affected metric instead
  of assigning zero; a valid empty transcription scores zero.
- Set model_task='image_generation' and eval_type='text2image'. For the official repeated-image
  protocol, use repeats=4; limit is intended only for smoke runs.


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
      "id": "e0b33751",
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
    "prompt_id": 0
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
