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
- Hugging Face remains available by setting `extra_params.dataset_hub="huggingface"`
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

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
Answer the question according to the image using a short phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes).
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_hub` | `str` | `modelscope` | Dataset hub used to load VQAv2 annotations and images. Choices: ['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | Source split to load; defaults to validation. |
| `dataset_revision` | `str` | `` | Optional dataset revision; leave empty to use the hub default. |
| `image_dir` | `str` | `` | Optional local directory containing VQAv2 images for local JSONL/CSV data. |
| `image_extension` | `str` | `` | Optional extension override for local images, for example "jpg". |

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
    dataset_args={
        'vqav2': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


