# SURDS


## Overview

SURDS benchmarks fine-grained spatial understanding and reasoning by vision-language models in realistic driving
scenes. It is derived from the six-camera nuScenes dataset and evaluates object-centric and relational spatial skills
without supplying depth maps or visual markers.

## Task Description

- **Task Type**: Multi-task visual spatial question answering
- **Input**: A 1600 x 900 driving-scene image and an English spatial reasoning question
- **Output**: A structured response ending in an answer inside `<answer>...</answer>`
- **Domain**: Autonomous driving and outdoor 3D spatial reasoning

## Key Features

- 9,250 model queries generated deterministically from 5,919 validation images, following the official seed-42 code
- Six equally weighted task subsets: yaw orientation, pixel localization, depth range, pairwise distance, left/right
  ordering, and front/behind relation
- Yaw, distance, left/right, and front/behind are consistency tests: both complementary prompts for an evaluation unit
  must be correct to receive credit
- Images come from six nuScenes cameras and contain unmarked objects described by appearance rather than overlays

## Evaluation Notes

- The official prompts and `<think>...<answer>...</answer>` response contract are reproduced verbatim
- Pixel localization uses the official centerness metric: predictions outside the target box receive 0, while points
  nearer the box center receive scores approaching 1; normalized coordinates and predicted boxes are also accepted
- The other five tasks use official normalized exact match, removing case, punctuation, articles, and extra whitespace
- Every subset contains 925 evaluation units; the overall normalized score is therefore the equal average of all six
  task scores. A full run makes 9,250 model requests but reports `Num=5,550`, because each complementary prompt pair
  is one official evaluation unit
- Invalid or missing `<answer>` blocks score 0, matching the official benchmark denominator semantics
- The dataset is evaluation-only and downloaded from ModelScope; only images needed by the selected subsets are fetched
- Resources: [Paper](https://arxiv.org/abs/2411.13112) |
  [GitHub](https://github.com/XiandaGuo/Drive-MLLM)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `surds` |
| **Dataset ID** | [evalscope/SURDS_eval](https://modelscope.cn/datasets/evalscope/SURDS_eval/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2411.13112) |
| **Tags** | `Grounding`, `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `normalized_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 9,250 |
| Prompt Length (Mean) | 728.31 chars |
| Prompt Length (Min/Max) | 631 / 910 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `yaw` | 1,850 | 725.18 | 714 | 769 |
| `xy2d` | 925 | 677.32 | 672 | 717 |
| `depth` | 925 | 861.75 | 854 | 901 |
| `distance` | 1,850 | 770.04 | 743 | 910 |
| `left_right` | 1,850 | 658.04 | 631 | 798 |
| `front_behind` | 1,850 | 718.77 | 703 | 791 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 9,250 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 1600x900 - 1600x900 |
| Formats | webp |


## Sample Example

**Subset**: `yaw`

```json
{
  "input": [
    {
      "id": "1e46d04d",
      "content": [
        {
          "text": "Task Description: \nThe primary goal of this task is to identify the direction that the specified object is facing in the given image. The camera in the image is facing North, and you need to analyze the object's orientation based on this refe ... [TRUNCATED 232 chars] ... evant error checks.\nFinally, provide a concise and definitive response in the <answer> tag. Use the following format:\n<think>[Step-by-step reasoning with attention to detail and potential error checks]</think>\n<answer>[Final answer]</answer>\n"
        },
        {
          "image": "~/.cache/modelscope/hub/datasets/evalscope/SURDS_eval/validation/image/CAM_BACK_RIGHT/nuscenes_0033_CAM_BACK_RIGHT.webp"
        }
      ]
    }
  ],
  "target": "West",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task": "yaw",
    "pair_id": "yaw-3",
    "variant_index": 0,
    "paired": true,
    "bbox": [
      662,
      504,
      774,
      545
    ],
    "options": [
      "North",
      "South",
      "East",
      "West"
    ],
    "image_size": [
      1600,
      900
    ],
    "image_path": "~/.cache/modelscope/hub/datasets/evalscope/SURDS_eval/validation/image/CAM_BACK_RIGHT/nuscenes_0033_CAM_BACK_RIGHT.webp"
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
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets surds \
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
    datasets=['surds'],
    dataset_args={
        'surds': {
            # subset_list: ['yaw', 'xy2d', 'depth']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
