# Ref-Adv-s


## Overview

Ref-Adv-s is the public 1,142-case subset of Ref-Adv, a referring expression comprehension benchmark designed to test whether multimodal large language models can distinguish a target from hard same-category visual distractors instead of relying on grounding shortcuts.

## Task Description

- **Task Type**: Referring expression comprehension / visual grounding
- **Input**: One image and an English referring expression
- **Output**: One or more bounding boxes in JSON, with the first box used for scoring
- **Domain**: COCO and OpenImages scenes containing hard same-category distractors

## Key Features

- Contains 1,142 public cases sampled from the 5,000-case Ref-Adv benchmark
- Includes human-authored and model-assisted expressions, explicit negation, and at least two distractors per case
- Preserves the official `direct` and chain-of-thought (`cot`) prompt modes
- Uses the dataset's single `train` split as the evaluation split

## Evaluation Notes

- Reports official `Acc@0.5`, `Acc@0.75`, and `Acc@0.9` metrics from the IoU of the first parsed box
- Also reports `Acc@0.5` for the official distractor-count bins `2-3`, `4-6`, and `>=7`
- Parses the last valid fenced JSON object, or an unfenced JSON value that ends the response, using the official key search order
- A failed first parse triggers the official one-turn format-repair prompt; a second failure receives zero accuracy
- Set `pred_box_format` to `abs_xyxy` for Qwen2.5-VL and to `norm_1000_xyxy` for Qwen3-VL/Qwen3.5; `norm_1_xyxy` is also supported by the official evaluator
- [Paper](https://arxiv.org/abs/2602.23898) | [GitHub](https://github.com/dddraxxx/Ref-Adv)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ref_adv_s` |
| **Dataset ID** | [evalscope/ref-adv-s](https://modelscope.cn/datasets/evalscope/ref-adv-s/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2602.23898) |
| **Tags** | `Grounding`, `MultiModal`, `Reasoning` |
| **Metrics** | `ACC@0.5`, `ACC@0.75`, `ACC@0.9`, `2-3/ACC@0.5`, `4-6/ACC@0.5`, `>=7/ACC@0.5` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,142 |
| Prompt Length (Mean) | 177.67 chars |
| Prompt Length (Min/Max) | 136 / 282 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,142 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 240x320 - 1024x1024 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d3a7e578",
      "content": [
        {
          "text": "<image>\nLocate every object that matches the description \"the computer screen that is in the middle vertically of the three stacks\" in the image. Report bbox coordinates in JSON format."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~118.1KB]"
        }
      ]
    }
  ],
  "target": "[297.0, 345.0, 427.0, 440.0]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "row_idx": 0,
    "file_name": "000000547144.jpg",
    "image_source": "coco_val2017",
    "human_authored": true,
    "use_negation": false,
    "distractor_count": 5,
    "target_box_normalized": [
      0.4640625,
      0.71875,
      0.6671875,
      0.9166666666666666
    ],
    "sent_size": [
      640,
      480
    ],
    "retry_followup_used": false
  }
}
```

## Prompt Template

**Prompt Template:**
```text
<image>
Locate every object that matches the description "{ref_sentence}" in the image. Report bbox coordinates in JSON format.
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_mode` | `str` | `direct` | Official prompt mode. Choices: ['direct', 'cot'] |
| `pred_box_format` | `str` | `norm_1000_xyxy` | Coordinate format emitted by the evaluated model. Choices: ['abs_xyxy', 'norm_1000_xyxy', 'norm_1_xyxy'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ref_adv_s \
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
    datasets=['ref_adv_s'],
    dataset_args={
        'ref_adv_s': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
