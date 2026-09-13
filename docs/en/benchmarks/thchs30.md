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
- Primary metric: Phone Error Rate (PER), computed as phone-token edit distance divided by reference phone count
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


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

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
