# RefCOCO


## Overview

RefCOCO is a dataset for training and evaluating models on Referring Expression Comprehension (REC). It contains images, object bounding boxes, and free-form natural-language expressions that uniquely describe target objects within MSCOCO images.

## Task Description

- **Task Type**: Referring Expression Comprehension / Image Captioning
- **Input**: Image (with visualization) + referring expression
- **Output**: Bounding box coordinates or caption
- **Domains**: Visual grounding, object localization, image understanding

## Key Features

- Created via Amazon Mechanical Turk annotations
- Three evaluation modes:
  - `bbox`: Image captioning with bounding box visualization
  - `seg`: Image captioning with segmentation visualization
  - `bbox_rec`: Grounding task - output normalized bounding box coordinates
- Expressions uniquely identify target objects in complex scenes
- Multiple subsets: test, val, testA, testB

## Evaluation Notes

- Evaluation mode configurable via `eval_mode` parameter
- Multiple metrics for comprehensive evaluation:
  - Grounding: IoU, ACC@0.1/0.3/0.5/0.7/0.9, Center_ACC
  - Captioning: BLEU (1-4), METEOR, ROUGE_L, CIDEr
- Bounding boxes output as normalized coordinates [x1/W, y1/H, x2/W, y2/H]
- Requires pycocoevalcap for caption metrics


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `refcoco` |
| **Dataset ID** | [lmms-lab/RefCOCO](https://modelscope.cn/datasets/lmms-lab/RefCOCO/summary) |
| **Paper** | N/A |
| **Tags** | `Grounding`, `ImageCaptioning`, `Knowledge`, `MultiModal` |
| **Metrics** | `IoU`, `ACC@0.1`, `ACC@0.3`, `ACC@0.5`, `ACC@0.7`, `ACC@0.9`, `Center_ACC`, `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `N/A` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 17,596 |
| Prompt Length (Mean) | 146 chars |
| Prompt Length (Min/Max) | 146 / 146 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `test` | 5,000 | 146 | 146 | 146 |
| `val` | 8,811 | 146 | 146 | 146 |
| `testA` | 1,975 | 146 | 146 | 146 |
| `testB` | 1,810 | 146 | 146 | 146 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 13,785 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 300x176 - 640x640 |
| Formats | jpeg |


## Sample Example

**Subset**: `test`

```json
{
  "input": [
    {
      "id": "53a494fc",
      "content": [
        {
          "text": "Please carefully observe the area circled in the image and come up with a caption for the area.\nAnswer the question using a single word or phrase."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~57.6KB]"
        }
      ]
    }
  ],
  "target": "['guy petting elephant', 'foremost person', 'green shirt']",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "469306",
    "iscrowd": 0,
    "file_name": "COCO_train2014_000000296747_0.jpg",
    "answer": [
      "guy petting elephant",
      "foremost person",
      "green shirt"
    ],
    "original_bbox": [
      59.04999923706055,
      93.23999786376953,
      375.0199890136719,
      362.5799865722656
    ],
    "bbox": [],
    "eval_mode": "bbox"
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `bbox` | Control the evaluation mode used by RefCOCO. bbox: image caption task, visualize the original image with bounding box; seg: image caption task, visualize the original image with segmentation; bbox_rec: grounding task, recognize bounding box coordinates. Choices: ['bbox', 'seg', 'bbox_rec'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets refcoco \
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
    datasets=['refcoco'],
    dataset_args={
        'refcoco': {
            # subset_list: ['test', 'val', 'testA']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


