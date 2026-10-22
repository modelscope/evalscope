# MMAU


## Overview

MMAU (Massive Multitask Audio Understanding) is a comprehensive benchmark for evaluating audio understanding capabilities of multimodal large language models across diverse audio tasks.

## Task Description

- **Task Type**: Audio Understanding (Multiple Choice)
- **Input**: Audio recordings with multiple-choice questions
- **Output**: Correct answer choice (A/B/C/D)
- **Domain**: Speech, Sound, Music

## Key Features

- Large-scale audio understanding benchmark
- Covers multiple audio domains (speech, environmental sounds, music)
- Multiple-choice format with 4 options
- Scores the `test_mini` split; the full `test` split ships without answers (`'NA'`)
- Per-category accuracy reporting

## Evaluation Notes

- Default configuration uses **test_mini** split
- Primary metric: **Accuracy** (exact match on predicted letter)
- Uses the last `ANSWER:` line containing only a single choice letter (case-insensitive, with optional brackets or Markdown bold); a bare choice letter is also accepted.
- Unparseable replies score zero. Natural-language answers are not mapped to choices.
- Evaluation version **v1.1** excludes the unlabeled `test` split and removes guessed answers.
- Reports overall accuracy and per-task-category accuracy
- Prompt includes chain-of-thought instruction
- Requires an OpenAI-compatible server that supports the `input_audio` content part for audio input.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mmau` |
| **Dataset ID** | [lmms-lab/mmau](https://modelscope.cn/datasets/lmms-lab/mmau/summary) |
| **Paper** | N/A |
| **Tags** | `Audio`, `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test_mini` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question based on the audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmau \
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
    datasets=['mmau'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
