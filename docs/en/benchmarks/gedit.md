# GEdit-Bench


## Overview

GEdit-Bench (Grounded Edit Benchmark) is an image editing benchmark grounded in real-world usage scenarios. It provides comprehensive evaluation of image editing models across diverse editing tasks with LLM-based judging.

## Task Description

- **Task Type**: Image Editing Evaluation
- **Input**: Source image + editing instruction
- **Output**: Edited image evaluated by LLM judge
- **Languages**: English (en) and Chinese (cn)

## Key Features

- Real-world editing scenarios (background change, color alter, style transfer, etc.)
- 11 editing task categories
- LLM-based evaluation for semantic consistency and perceptual quality
- Supports both English and Chinese instructions
- Comprehensive scoring: Semantic Consistency, Perceptual Quality, Overall

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **train** split (contains test samples)
- Metrics: **Semantic Consistency**, **Perceptual Similarity** (via LLM judge)
- Overall score: geometric mean of SC and PQ scores
- Configure language via `extra_params['language']` (en/cn)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gedit` |
| **Dataset ID** | [stepfun-ai/GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench/summary) |
| **Paper** | N/A |
| **Tags** | `ImageEditing` |
| **Metrics** | `Semantic Consistency`, `Perceptual Similarity` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 606 |
| Prompt Length (Mean) | 42.46 chars |
| Prompt Length (Min/Max) | 11 / 158 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `background_change` | 40 | 50.2 | 29 | 158 |
| `color_alter` | 40 | 41.5 | 23 | 143 |
| `material_alter` | 40 | 40.8 | 18 | 60 |
| `motion_change` | 40 | 44.05 | 20 | 87 |
| `ps_human` | 70 | 34.17 | 16 | 89 |
| `style_change` | 60 | 46.27 | 20 | 116 |
| `subject-add` | 60 | 51.13 | 14 | 148 |
| `subject-remove` | 57 | 37.3 | 15 | 110 |
| `subject-replace` | 60 | 48.95 | 27 | 96 |
| `text_change` | 99 | 39.71 | 11 | 116 |
| `tone_transfer` | 40 | 36 | 21 | 63 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 606 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 384x640 - 416x672 |
| Formats | png |


## Sample Example

**Subset**: `background_change`

```json
{
  "input": [
    {
      "id": "4c309b59",
      "content": [
        {
          "text": "Change the background to a city street."
        },
        {
          "image": "[BASE64_IMAGE: png, ~495.7KB]"
        }
      ]
    }
  ],
  "id": 0,
  "group_id": 0,
  "subset_key": "background_change",
  "metadata": {
    "task_type": "background_change",
    "key": "4a7d36259ad94d238a6e7e7e0bd6b643",
    "instruction": "Change the background to a city street.",
    "instruction_language": "en",
    "input_image": "[BASE64_IMAGE: png, ~495.7KB]",
    "Intersection_exist": true,
    "id": "4a7d36259ad94d238a6e7e7e0bd6b643"
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `en` | Language of the instruction. Choices: ['en', 'cn']. Choices: ['en', 'cn'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gedit \
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
    datasets=['gedit'],
    dataset_args={
        'gedit': {
            # subset_list: ['background_change', 'color_alter', 'material_alter']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


