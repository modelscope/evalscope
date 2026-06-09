# VQAv2


## Overview

VQAv2 is the balanced Visual Question Answering benchmark built on COCO images. It evaluates whether
multimodal models can answer open-ended natural-language questions grounded in image content.

## Task Description

- **Task Type**: Open-ended visual question answering
- **Input**: Image + natural-language question
- **Output**: Short answer phrase
- **Domains**: General image understanding, object recognition, counting, attributes, relations

## Evaluation Notes

- Default data source: `lmms-lab/VQAv2` on ModelScope, `validation` split
- Primary metric: **VQAv2 soft accuracy** over human annotator answers
- Also reports normalized exact match against the available answer set
- The adapter accepts common answer formats: list of strings, list of answer dicts, or `multiple_choice_answer`


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `vqav2` |
| **Dataset ID** | [lmms-lab/VQAv2](https://modelscope.cn/datasets/lmms-lab/VQAv2/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/1612.00837) |
| **Tags** | `MultiModal`, `QA` |
| **Metrics** | `vqa_score`, `exact_match` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 214,354 |
| Prompt Length (Mean) | 185.83 chars |
| Prompt Length (Min/Max) | 165 / 255 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 214,354 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 120x120 - 640x640 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "7410f0b5",
      "content": [
        {
          "text": "Answer the question according to the image using a short phrase.\nWhere is he looking?\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes)."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~102.7KB]"
        }
      ]
    }
  ],
  "target": "[\"down\", \"down\", \"at table\", \"skateboard\", \"down\", \"table\", \"down\", \"down\", \"down\", \"down\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question": "Where is he looking?",
    "answers": [
      "down",
      "down",
      "at table",
      "skateboard",
      "down",
      "table",
      "down",
      "down",
      "down",
      "down"
    ],
    "multiple_choice_answer": "down",
    "question_id": 262148000,
    "question_type": "none of the above",
    "answer_type": "other"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the question according to the image using a short phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes).
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vqav2 \
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
    datasets=['vqav2'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


