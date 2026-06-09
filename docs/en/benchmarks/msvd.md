# MSVD


## Overview

MSVD is a classic video captioning benchmark with short web videos annotated by many human captions.
The native adapter treats each video as one evaluation sample and uses all available captions as references.

## Task Description

- **Task Type**: Video captioning
- **Input**: Video clip
- **Output**: One concise natural-language caption
- **Domains**: Open-domain video understanding and description

## Evaluation Notes

- Default data source: `evalscope/MSVD` on ModelScope, `test` split
- Hugging Face `VLM2Vec/MSVD` remains available by setting `extra_params.dataset_hub="huggingface"`
- Primary metric: **CIDEr**
- Additional metrics: BLEU-1/2/3/4, METEOR, ROUGE-L
- Set `extra_params.video_dir` when the dataset only provides video file names and local media files are required


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `msvd` |
| **Dataset ID** | [evalscope/MSVD](https://modelscope.cn/datasets/evalscope/MSVD/summary) |
| **Paper** | [Paper](https://aclanthology.org/P11-1020/) |
| **Tags** | `ImageCaptioning`, `MultiModal` |
| **Metrics** | `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 670 |
| Prompt Length (Mean) | 43 chars |
| Prompt Length (Min/Max) | 43 / 43 chars |

**Video Statistics:**

| Metric | Value |
|--------|-------|
| Total Videos | 670 |
| Videos per Sample | min: 1, max: 1, mean: 1 |
| Formats | mp4 |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "a4b83275",
      "content": [
        {
          "text": "Describe the video in one concise sentence."
        },
        {
          "video": "fr9H1WLcF1A_256_261.avi",
          "format": "mp4"
        }
      ]
    }
  ],
  "target": "[\"two young men are playing table tennis\", \"men are playing table tennis\", \"two men are playing table tennis\", \"two men are playing a tabletennis\", \"a couple of people are playing a game of ping pong\", \"peoples are playing table tennis\", \"two ... [TRUNCATED 306 chars] ... e boys are playing\", \"people are playing ping pong\", \"18 kids and counting show the newest baby josie\", \"there is some kids and playing to each other\", \"the men played pingpong together\", \"the boys are playing ping pong\", \"2 boys is playing\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "references": [
      "two young men are playing table tennis",
      "men are playing table tennis",
      "two men are playing table tennis",
      "two men are playing a tabletennis",
      "a couple of people are playing a game of ping pong",
      "peoples are playing table tennis",
      "two men are playing pingpong",
      "two guys play table tennis",
      "two people are playing ping pong",
      "two boys are playing ping pong",
      "... [TRUNCATED 12 more items] ..."
    ],
    "subset": "default",
    "dataset_id": "evalscope/MSVD",
    "dataset_hub": "modelscope",
    "video": "fr9H1WLcF1A_256_261.avi",
    "video_id": "fr9H1WLcF1A_256_261",
    "source": "MSVD"
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
| `dataset_hub` | `str` | `modelscope` | Dataset hub used to load MSVD annotations. Choices: ['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | Source split to load; defaults to test. |
| `dataset_revision` | `str` | `` | Optional dataset revision; leave empty to use the hub default. |
| `video_dir` | `str` | `` | Optional local directory containing MSVD video files. |
| `video_extension` | `str` | `` | Optional extension override for local videos, for example "mp4". |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets msvd \
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
    datasets=['msvd'],
    dataset_args={
        'msvd': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


