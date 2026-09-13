# LibriSpeech


## Overview

LibriSpeech is a large-scale corpus of approximately 1,000 hours of read English speech derived from audiobooks. It is one of the most widely used benchmarks for evaluating automatic speech recognition (ASR) systems.

## Task Description

- **Task Type**: Automatic Speech Recognition (ASR)
- **Input**: Audio recordings of read English speech from audiobooks
- **Output**: Transcribed text
- **Language**: English

## Key Features

- 1,000 hours of high-quality read speech
- Derived from LibriVox audiobooks (public domain)
- Clean and "other" test sets for varying difficulty
- Widely used baseline for ASR research
- Standardized evaluation protocol

## Evaluation Notes

- Supported evaluation splits: **test_clean** and **test_other**
- Primary metric: **Word Error Rate (WER)**
- Text normalization applied during evaluation
- Prompt: "Please recognize the speech and only output the recognized content"
- Metadata includes audio ID and duration information


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `librispeech` |
| **Dataset ID** | [lmms-lab/Librispeech-concat](https://modelscope.cn/datasets/lmms-lab/Librispeech-concat/summary) |
| **Paper** | N/A |
| **Tags** | `Audio`, `SpeechRecognition` |
| **Metrics** | `wer` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `N/A` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 177 |
| Prompt Length (Mean) | 67 chars |
| Prompt Length (Min/Max) | 67 / 67 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `test_clean` | 87 | 67 | 67 | 67 |
| `test_other` | 90 | 67 | 67 | 67 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 177 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | wav |


## Sample Example

**Subset**: `test_clean`

```json
{
  "input": [
    {
      "id": "732c90a4",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~22.7MB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "Eleven o'clock had struck it was a fine clear night they were the only persons on the road and they sauntered leisurely along to avoid paying the price of fatigue for the recreation provided for the toledans in their valley or on the banks of ... [TRUNCATED 7380 chars] ...  less surprised than they and the better to assure himself of so wonderful a fact he begged leocadia to give him some token which should make perfectly clear to him that which indeed he did not doubt since it was authenticated by his parents.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "audio_id": "5639-40744",
    "audio_duration": 496.6899719238281
  }
}
```

*Note: Some content was truncated for display.*

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
    --datasets librispeech \
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
    datasets=['librispeech'],
    dataset_args={
        'librispeech': {
            # subset_list: ['test_clean', 'test_other']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
