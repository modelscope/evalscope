# MSR-VTT


## Overview

MSR-VTT is a large-scale open-domain video captioning benchmark for evaluating video-to-text generation.
The native adapter groups records by `video_id`, so multiple annotation rows for one video become one sample
with multiple reference captions.

## Task Description

- **Task Type**: Video captioning
- **Input**: Video clip or URL
- **Output**: One concise natural-language caption
- **Domains**: Open-domain video understanding and description

## Evaluation Notes

- Default data source: `AI-ModelScope/msr-vtt` on ModelScope, `validation` split
- Hugging Face `VLM2Vec/MSR-VTT` remains available by setting `extra_params.dataset_hub="huggingface"`
- Primary metric: **CIDEr**
- Additional metrics: BLEU-1/2/3/4, METEOR, ROUGE-L
- Set `extra_params.video_dir` to prefer local media files over URL metadata


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `msr_vtt` |
| **Dataset ID** | [AI-ModelScope/msr-vtt](https://modelscope.cn/datasets/AI-ModelScope/msr-vtt/summary) |
| **Paper** | [Paper](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) |
| **Tags** | `ImageCaptioning`, `MultiModal` |
| **Metrics** | `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 497 |
| Prompt Length (Mean) | 43 chars |
| Prompt Length (Min/Max) | 43 / 43 chars |

**Video Statistics:**

| Metric | Value |
|--------|-------|
| Total Videos | 497 |
| Videos per Sample | min: 1, max: 1, mean: 1 |
| Formats | mp4 |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "36044e4b",
      "content": [
        {
          "text": "Describe the video in one concise sentence."
        },
        {
          "video": "https://www.youtube.com/watch?v=A9pM9iOuAzM",
          "format": "mp4",
          "start": 116.03,
          "end": 126.21
        }
      ]
    }
  ],
  "target": "[\"a family is having coversation\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "references": [
      "a family is having coversation"
    ],
    "subset": "default",
    "dataset_id": "AI-ModelScope/msr-vtt",
    "dataset_hub": "modelscope",
    "video": "https://www.youtube.com/watch?v=A9pM9iOuAzM",
    "start": 116.03,
    "end": 126.21,
    "fps": null,
    "video_id": "video6513",
    "category": 14
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Describe the video in one concise sentence.
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_hub` | `str` | `modelscope` | Dataset hub used to load MSR-VTT annotations. Choices: ['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | Source split to load; defaults to validation for ModelScope and test for Hugging Face. |
| `dataset_revision` | `str` | `` | Optional dataset revision; leave empty to use the hub default. |
| `video_dir` | `str` | `` | Optional local directory containing MSR-VTT video files. |
| `video_extension` | `str` | `` | Optional extension override for local videos, for example "mp4". |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets msr_vtt \
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
    datasets=['msr_vtt'],
    dataset_args={
        'msr_vtt': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


