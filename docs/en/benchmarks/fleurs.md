# FLEURS


## Overview

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a massively multilingual benchmark covering 102 languages for evaluating automatic speech recognition (ASR), spoken language understanding, and speech translation.

## Task Description

- **Task Type**: Automatic Speech Recognition (ASR)
- **Input**: Audio recordings with speech in various languages
- **Output**: Transcribed text in the corresponding language
- **Languages**: 102 languages including Mandarin Chinese, Cantonese, English, and many more

## Key Features

- Massive multilingual coverage (102 languages)
- Derived from FLoRes-101 machine translation benchmark
- Includes diverse language families and scripts
- High-quality human recordings and transcriptions
- Metadata includes gender, language group, and speaker information

## Evaluation Notes

- Default configuration uses **test** split
- Primary metric: **Word Error Rate (WER)**
- Default subsets: `cmn_hans_cn` (Mandarin), `en_us` (English), `yue_hant_hk` (Cantonese)
- Language-specific text normalization applied during evaluation
- Prompt: "Please recognize the speech and only output the recognized content"


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `fleurs` |
| **Dataset ID** | [lmms-lab/fleurs](https://modelscope.cn/datasets/lmms-lab/fleurs/summary) |
| **Paper** | N/A |
| **Tags** | `Audio`, `MultiLingual`, `SpeechRecognition` |
| **Metrics** | `wer` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,411 |
| Prompt Length (Mean) | 67 chars |
| Prompt Length (Min/Max) | 67 / 67 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `cmn_hans_cn` | 945 | 67 | 67 | 67 |
| `en_us` | 647 | 67 | 67 | 67 |
| `yue_hant_hk` | 819 | 67 | 67 | 67 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 2,411 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | wav |


## Sample Example

**Subset**: `cmn_hans_cn`

```json
{
  "input": [
    {
      "id": "daf508c3",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~648.8KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束 也 是 新 篇 章 的 开 始",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": 1906,
    "num_samples": 166080,
    "raw_transcription": "“这并不是告别。这是一个篇章的结束，也是新篇章的开始。”",
    "language": "Mandarin Chinese",
    "gender": 0,
    "lang_id": "cmn_hans",
    "lang_group_id": 6
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
    --datasets fleurs \
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
    datasets=['fleurs'],
    dataset_args={
        'fleurs': {
            # subset_list: ['cmn_hans_cn', 'en_us', 'yue_hant_hk']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


