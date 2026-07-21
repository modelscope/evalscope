# EmbSpatial-Bench


## Overview

EmbSpatial-Bench is a benchmark for evaluating embodied spatial understanding of large vision-language models (LVLMs). The benchmark is automatically derived from embodied scenes and covers 6 spatial relationships from an egocentric perspective: **close**, **far**, **above**, **under**, **left**, and **right**.

## Task Description

- **Task Type**: Multiple-Choice Visual Question Answering (VQA)
- **Input**: An egocentric RGB image + a spatial reasoning question with 4 candidate answers
- **Output**: A single letter (A / B / C / D) identifying the correct object or spatial relationship
- **Domains**: Embodied AI, spatial reasoning (MP3D and AI2Thor environments)

## Key Features

- 3,640 human-verified evaluation questions derived from two embodied environments (MP3D and AI2Thor)
- 6 spatial relation categories: close, far, above, under, left, right
- Each question requires selecting the most spatially accurate answer from 4 options
- Designed to expose the gap between current LVLMs and qualified embodied intelligence

## Evaluation Notes

- Default evaluation uses the **embspatial_bench.json** file (3,640 samples)
- Primary metric: **Accuracy** (acc)
- Answer indices are 0-based in the dataset (0 → A, 1 → B, 2 → C, 3 → D)
- Images are stored as JPEG base64 strings in the JSON file
- Subsets are organized by the `relation` field (6 spatial categories)
- [Paper](https://aclanthology.org/2024.acl-short.33/) | [GitHub](https://github.com/mengfeidu/EmbSpatial-Bench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `emb_spatial_bench` |
| **Dataset ID** | [evalscope/EmbSpatial-Bench](https://modelscope.cn/datasets/evalscope/EmbSpatial-Bench/summary) |
| **Paper** | [Paper](https://aclanthology.org/2024.acl-short.33/) |
| **Tags** | `MCQ`, `MultiModal`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,640 |
| Prompt Length (Mean) | 369.34 chars |
| Prompt Length (Min/Max) | 265 / 490 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `close` | 612 | 293.54 | 274 | 326 |
| `far` | 594 | 292.68 | 265 | 326 |
| `above` | 596 | 405.61 | 360 | 486 |
| `under` | 602 | 404.35 | 350 | 490 |
| `left` | 616 | 408.86 | 359 | 486 |
| `right` | 620 | 409.47 | 352 | 481 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 3,640 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 300x300 - 1296x968 |
| Formats | jpeg |


## Sample Example

**Subset**: `close`

```json
{
  "input": [
    {
      "id": "3f406c29",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~35.2KB]"
        },
        {
          "text": "Among the listed objects, which one is closest to your current location in the image?\n(A) table\n(B) towel\n(C) door\n(D) basket\nAnswer with only the letter of the correct option. The last line of your response should be of the format: ANSWER: [LETTER] where LETTER is one of A, B, C, D."
        }
      ]
    }
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "close",
  "metadata": {
    "question_id": "mp3d_0",
    "relation": "close",
    "data_source": "mp3d"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets emb_spatial_bench \
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
    datasets=['emb_spatial_bench'],
    dataset_args={
        'emb_spatial_bench': {
            # subset_list: ['close', 'far', 'above']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
