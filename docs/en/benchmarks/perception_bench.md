# PerceptionBench


## Overview

PerceptionBench is a benchmark from Moonshot AI that evaluates the atomic visual perception
capabilities of multimodal large language models. It is built bottom-up: the earliest failure
points of frontier MLLMs on 42 existing benchmarks were diagnosed to derive an error taxonomy
whose perception branch defines ten atomic perceptual capabilities. Each question isolates a
single capability, so difficulty stems from perception rather than reasoning or knowledge.

## Task Description

- **Task Type**: Visual Perception (open-ended question answering)
- **Input**: One or more images interleaved with a question
- **Output**: Free-form short answer with a uniquely determined reference
- **Domain**: Atomic visual perception across ten capabilities

## Key Features

- 3,000 verified questions covering ten atomic perceptual capabilities
- 1,800 questions (60%) are atomic sub-questions decomposed from attributed failures on source
  benchmarks; 1,200 (40%) are newly authored on supplemented images
- Subsets follow the ten `error_category` labels: visual relation, counting, attribute,
  depth & 3D perception, localization, comparison, fine-grained recognition, contextual
  integration, OCR, and perception-related hallucination
- Multi-image questions are supported: images are interleaved into the question via
  `<|image_N|>` placeholders
- Samples carrying a `hint` (coordinate convention or image dimensions) pass it as a system
  message, matching the official message builder

## Evaluation Notes

- Default evaluation uses the **train** split (3,000 samples, single split dataset)
- Primary metric: **Accuracy**, reported overall and per capability
- Scoring follows the official protocol: an LLM judge grades the free-form answer against the
  reference with the teacher-grading prompt and returns a strict 0/1 verdict per item
  (`[reason]` / `[judge] True|False`); the paper uses GPT-oss-120B, whose agreement with human
  judgment is 99.7% on a 300-sample audit
- Empty or failed generations are scored 0 without invoking the judge
- Requires `judge_model_args` configuration for the LLM judge
- The dataset embeds images as base64 data URIs (~1.6 GB download on first use)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `perception_bench` |
| **Dataset ID** | [moonshotai/PerceptionBench](https://modelscope.cn/datasets/moonshotai/PerceptionBench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2607.24957) |
| **Tags** | `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,000 |
| Prompt Length (Mean) | 233.87 chars |
| Prompt Length (Min/Max) | 29 / 1076 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `visual_relation_error` | 330 | 275.62 | 43 | 876 |
| `visual_counting_error` | 330 | 161.11 | 37 | 831 |
| `visual_attribute_error` | 330 | 225.58 | 34 | 1006 |
| `depth_3d_perception_error` | 330 | 278.5 | 60 | 976 |
| `visual_localization_error` | 330 | 284.79 | 62 | 1076 |
| `visual_comparison_error` | 279 | 270.14 | 39 | 801 |
| `fine_grained_recognition_error` | 290 | 225.91 | 44 | 917 |
| `context_integration_error` | 255 | 277.04 | 58 | 845 |
| `ocr_error` | 255 | 175.39 | 29 | 934 |
| `hallucination` | 271 | 150.94 | 42 | 515 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 3,567 |
| Images per Sample | min: 1, max: 8, mean: 1.19 |
| Resolution Range | 101x64 - 5712x4953 |
| Formats | jpeg, png, webp |


## Sample Example

**Subset**: `visual_relation_error`

```json
{
  "input": [
    {
      "id": "28bf28ec",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~97.9KB]"
        },
        {
          "text": "How many arrows does the dashed box intersect with? Just answer with the number."
        }
      ]
    }
  ],
  "target": "4",
  "id": 0,
  "group_id": 0,
  "subset_key": "visual_relation_error",
  "metadata": {
    "index": 5,
    "problem": "<|image_1|>How many arrows does the dashed box intersect with? Just answer with the number.",
    "error_category": "visual_relation_error",
    "source_bmk": "NA",
    "source_idx": null
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
    --datasets perception_bench \
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
    datasets=['perception_bench'],
    dataset_args={
        'perception_bench': {
            # subset_list: ['visual_relation_error', 'visual_counting_error', 'visual_attribute_error']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
