# OmniBench


## Overview

OmniBench is a pioneering universal multimodal benchmark designed to rigorously evaluate MLLMs' capability to recognize, interpret, and reason across visual, acoustic, and textual inputs simultaneously.

## Task Description

- **Task Type**: Universal Multimodal Understanding (Image + Audio + Text)
- **Input**: Image, audio, and text question with choices
- **Output**: Correct answer choice
- **Modalities**: Visual, acoustic, and textual

## Key Features

- Tri-modal evaluation (image, audio, text)
- Tests cross-modal reasoning abilities
- Multiple-choice format
- Configurable modality usage
- Chain-of-thought prompting

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Simple accuracy metric
- Extra params: use_image (bool), use_audio (bool)
- Supports textual alternatives for image/audio


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `omni_bench` |
| **Dataset ID** | [m-a-p/OmniBench](https://modelscope.cn/datasets/m-a-p/OmniBench/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,142 |
| Prompt Length (Mean) | 477.61 chars |
| Prompt Length (Min/Max) | 310 / 1280 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,142 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 361x203 - 5184x3456 |
| Formats | jpeg, png |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 1,142 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | mp3 |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "a5cb4fd0",
      "content": [
        {
          "text": "Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhat are the men doing?\n\nA) The man in jeans is taking notes from the newspaper.\nB) The man in purple is reading the newspaper.\nC) The man in jeans is playing a crossword puzzle.\nD) The man on the table is doing a crossword puzzle."
        },
        {
          "image": "[BASE64_IMAGE: png, ~3.7MB]"
        },
        {
          "audio": "[BASE64_AUDIO: mp3, ~103.6KB]",
          "format": "mp3"
        }
      ]
    }
  ],
  "choices": [
    "The man in jeans is taking notes from the newspaper.",
    "The man in purple is reading the newspaper.",
    "The man in jeans is playing a crossword puzzle.",
    "The man on the table is doing a crossword puzzle."
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "task_type": "Action and Activity",
    "audio_type": "speech",
    "answer": "The man in jeans is playing a crossword puzzle.",
    "image_content": "The image depicts a scene from a television show set in a cozy, well-furnished apartment. The main focus is on two individuals sitting on a couch in the foreground. The man on the left is wearing a dark blue jacket and jeans, and he is holdin ... [TRUNCATED] ... ted with a mix of modern and vintage elements, including a colorful rug, patterned curtains, and a variety of decorative items on the shelves and walls. The overall atmosphere is warm and inviting, suggesting a comfortable and lived-in space.",
    "audio_content": "Two people talking: A: Four letters, \"circle or hoop.\" B: Ring! Damn it, ring! A: Thanks."
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_image` | `bool` | `True` | Whether to provide the raw image. False uses textual alternative. |
| `use_audio` | `bool` | `True` | Whether to provide the raw audio. False uses textual alternative. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_bench \
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
    datasets=['omni_bench'],
    dataset_args={
        'omni_bench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


