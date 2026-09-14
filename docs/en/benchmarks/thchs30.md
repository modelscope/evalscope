# THCHS-30


## Overview

THCHS-30 is a Mandarin Chinese read-speech corpus with phone-level time alignments. This EvalScope benchmark evaluates phoneme recognition from speech using the aligned IPA phone sequences released with the dataset.

## Task Description

- **Task Type**: Phoneme Recognition
- **Input**: A 16 kHz Mandarin Chinese speech recording
- **Output**: A space-separated sequence of IPA phone tokens with tone diacritics
- **Domain**: Read Mandarin Chinese speech

## Key Features

- 13,388 utterances split into 10,000 train, 893 validation, and 2,495 test samples
- Phone-level start and end timestamps, including `[SIL]` silence intervals
- Original Hanzi transcripts, speaker identifiers, durations, and split labels retained as sample metadata
- Test evaluation uses the release's IPA phone inventory and tone annotations

## Evaluation Notes

- Default configuration evaluates the `test` split only; no few-shot examples are used
- Primary metric: corpus-level Phone Error Rate (PER), computed as total phone-token edit distance divided by total reference phone count; report counts remain utterance counts
- The model must output only IPA phone tokens separated by spaces; timestamps are metadata for downstream analysis, not model targets
- Audio is passed as WAV data through EvalScope's standard audio message format


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `thchs30` |
| **Dataset ID** | [evalscope/THCHS-30](https://modelscope.cn/datasets/evalscope/THCHS-30/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/1512.01882) |
| **Tags** | `Audio`, `Chinese`, `SpeechRecognition` |
| **Metrics** | `per` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `weighted_mean` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,495 |
| Prompt Length (Mean) | 88 chars |
| Prompt Length (Min/Max) | 88 / 88 chars |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 2,495 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | wav |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "307a05d8",
      "content": [
        {
          "text": "Transcribe the audio as IPA phone tokens. Output only tokens separated by single spaces."
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~284.9KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "[SIL] t a˥ ŋ ɻ a˧˥ː n [SIL] tʰ˘ a˥˘ m˘ ə n˘ p u˥˩˘ ɕ j a˥˩˘ ŋ ɤ˥˩ː y˧˥ n a˥˩ jˑ a˥˩˘ ŋ˘ tʰ w˘ ə˥˘ n ɕ˘ j a˥˩ ʂː ɻ̩˧˥ kʰ w ai̯˥˩ˑ [SIL] t a˥ ŋ ts˘ wˑ o˥˩ˑ jː a˥ tsʰ a˥ ŋ ʂː ɻ̩˧˥ˑ [SIL] ɚ˧˥˘ ʂ ɻ̩˥˩ mˑ w o˧˥ ʈʂʰ˘ ə˧˥ ŋ f ə˧˩˧ nː i˧˩˧ xː ou̯˥˩ tsʰ˘ ai̯˧˥˘ fˑ u˧˥ˑ j ʊ˥˩ ŋ [SIL]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "utt_id": "D11_779",
    "text": "当然 他们 不 像 鳄鱼 那样 吞下 石块 当做 压 舱 石 而是 磨成 粉 以后 才 服用",
    "phones": [
      "[SIL]",
      "t",
      "a˥",
      "ŋ",
      "ɻ",
      "a˧˥ː",
      "n",
      "[SIL]",
      "tʰ˘",
      "a˥˘",
      "... [TRUNCATED 66 more items] ..."
    ],
    "phone_starts": [
      0.0,
      1.55,
      1.6,
      1.69,
      1.75,
      1.81,
      1.96,
      2.03,
      2.33,
      2.39,
      "... [TRUNCATED 66 more items] ..."
    ],
    "phone_ends": [
      1.55,
      1.6,
      1.69,
      1.75,
      1.81,
      1.96,
      2.03,
      2.33,
      2.39,
      2.43,
      "... [TRUNCATED 66 more items] ..."
    ],
    "language": "cmn",
    "speaker_id": "D11",
    "duration": 9.114,
    "split": "test"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Transcribe the audio as IPA phone tokens. Output only tokens separated by single spaces.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets thchs30 \
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
    datasets=['thchs30'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
