# WenetSpeech


## Overview

WenetSpeech is a large-scale Mandarin Chinese speech corpus with over 10,000 hours of multi-domain transcribed audio data, designed for speech recognition research.

## Task Description

- **Task Type**: Automatic Speech Recognition (ASR)
- **Input**: Audio recordings with Mandarin Chinese speech
- **Output**: Transcribed text in Chinese
- **Domain**: Multi-domain (internet, meeting)

## Key Features

- Large-scale Mandarin Chinese speech corpus (10,000+ hours)
- Multi-domain coverage: internet content, meetings
- High-quality transcriptions
- Suitable for evaluating Chinese ASR systems
- Supports mixed Chinese-English text evaluation

## Evaluation Notes

- Default configuration uses **test_meeting** split
- Subsets by domain: **dev** (development), **test_meeting** (meeting domain)
- Primary metric: **MER** (Mixed Error Rate)
- MER tokenizes Chinese characters individually and English words as whole tokens
- Prompt: "Please listen to the audio and transcribe what you hear"


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `wenet_speech` |
| **Dataset ID** | [lmms-lab/WenetSpeech](https://modelscope.cn/datasets/lmms-lab/WenetSpeech/summary) |
| **Paper** | N/A |
| **Tags** | `Audio`, `SpeechRecognition` |
| **Metrics** | `mer` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test_meeting` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 22,195 |
| Prompt Length (Mean) | 161 chars |
| Prompt Length (Min/Max) | 161 / 161 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `dev` | 13,825 | 161 | 161 | 161 |
| `test_meeting` | 8,370 | 161 | 161 | 161 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 22,195 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | wav |


## Sample Example

**Subset**: `dev`

```json
{
  "input": [
    {
      "id": "c30c80b4",
      "content": [
        {
          "text": "Please listen to the audio and transcribe what you hear. Please only provide the transcription without any additional commentary. Do not include any punctuation."
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~175.3KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "text": "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Please listen to the audio and transcribe what you hear. Please only provide the transcription without any additional commentary. Do not include any punctuation.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets wenet_speech \
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
    datasets=['wenet_speech'],
    dataset_args={
        'wenet_speech': {
            # subset_list: ['dev', 'test_meeting']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


