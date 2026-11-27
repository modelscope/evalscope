# CVTG-2K


## Overview

CVTG-2K evaluates whether text-to-image models accurately render multiple text regions in one image.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Prompt specifying 2-5 quoted text regions
- **Output**: Image containing the requested text
- **Domain**: Multi-region text rendering

## Key Features

- 2,000 prompts from TextCrafter: CVTG (1,002) and CVTG-Style (998).
- CVTG-Style additionally specifies text size, color and font.
- Loads the ModelScope mirror by default; the official Hugging Face source is also supported.

## Evaluation Notes

- Install `evalscope[cvtg_2k]` for PaddleOCR scoring. Use Python 3.11 or 3.12 on a platform
  supported by PaddlePaddle 2.6.
- Word Accuracy and word-level normalized edit similarity (NED, higher is better) follow the official evaluator,
  aggregating across all ground-truth words rather than averaging image scores.
- These metrics assess text content, not font, color, layout or image aesthetics.
- Scoring failures are excluded; valid images with no recognized text count as missing text.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cvtg_2k` |
| **Dataset ID** | [evalscope/CVTG-2K](https://modelscope.cn/datasets/evalscope/CVTG-2K/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2503.23461) |
| **Tags** | `TextToImage` |
| **Metrics** | `CVTGWordAccuracy`, `CVTGNED` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,000 |
| Prompt Length (Mean) | 195.92 chars |
| Prompt Length (Min/Max) | 67 / 462 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `CVTG` | 1,002 | 161.05 | 67 | 343 |
| `CVTG-Style` | 998 | 230.93 | 106 | 462 |

## Sample Example

**Subset**: `CVTG`

```json
{
  "input": [
    {
      "id": "58a209ac",
      "content": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "prompt": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'.",
    "carrier_list": [
      "sign",
      "cup"
    ],
    "sentence_list": [
      "a sign that says 'READ'.",
      "a coffee cup on the table with the word 'MORNING'."
    ],
    "benchmark_type": "CVTG",
    "region_count": 2,
    "category": "CVTG",
    "tags": [],
    "id": "",
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
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets cvtg_2k \
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
    datasets=['cvtg_2k'],
    dataset_args={
        'cvtg_2k': {
            # subset_list: ['CVTG', 'CVTG-Style']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
