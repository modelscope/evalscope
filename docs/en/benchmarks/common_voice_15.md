# CommonVoice15


## Overview

Common Voice 15 is a massively multilingual speech corpus collected by Mozilla, covering 114 languages with thousands of hours of validated speech data from volunteers worldwide.

## Task Description

- **Task Type**: Automatic Speech Recognition (ASR)
- **Input**: Audio recordings with speech in various languages
- **Output**: Transcribed text in the corresponding language
- **Languages**: 114 languages including English, Mandarin Chinese, French, and many more

## Key Features

- Crowd-sourced speech recordings with community validation
- Diverse speaker demographics (age, gender, accent)
- Multiple languages with varying amounts of data
- CC-0 licensed for open research and commercial use
- High-quality transcriptions validated by multiple listeners

## Evaluation Notes

- Default configuration uses **test** split
- Primary metric: **Word Error Rate (WER)**
- Default subsets: `en` (English), `zh-CN` (Mandarin Chinese), `fr` (French)
- Language-specific text normalization applied during evaluation
- Prompt: "Please recognize the speech and only output the recognized content"


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `common_voice_15` |
| **Dataset ID** | [lmms-lab/common_voice_15](https://modelscope.cn/datasets/lmms-lab/common_voice_15/summary) |
| **Paper** | N/A |
| **Tags** | `Audio`, `MultiLingual`, `SpeechRecognition` |
| **Metrics** | `wer` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 43,143 |
| Prompt Length (Mean) | 67 chars |
| Prompt Length (Min/Max) | 67 / 67 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en` | 16,386 | 67 | 67 | 67 |
| `zh-CN` | 10,625 | 67 | 67 | 67 |
| `fr` | 16,132 | 67 | 67 | 67 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 43,143 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | mp3 |


## Sample Example

**Subset**: `en`

```json
{
  "input": [
    {
      "id": "88959854",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: mp3, ~37.0KB]",
          "format": "mp3"
        }
      ]
    }
  ],
  "target": "Joe Keaton disapproved of films, and Buster also had reservations about the medium.",
  "id": 0,
  "group_id": 0,
  "subset_key": "en",
  "metadata": {
    "locale": "en",
    "path": "/home/tiger/.cache/huggingface/datasets/downloads/extracted/f54628fae82dd952031cdea3ec9c3d600c11d606e00cb8b3fd1b6ad500d7eb23/en_test_0/common_voice_en_27710027.mp3",
    "lang_id": "en"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Please recognize the speech and only output the recognized content:
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets common_voice_15 \
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
    datasets=['common_voice_15'],
    dataset_args={
        'common_voice_15': {
            # subset_list: ['en', 'zh-CN', 'fr']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
