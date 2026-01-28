# MRI-MCQA

## Overview

MRI-MCQA is a specialized benchmark composed of multiple-choice questions related to Magnetic Resonance Imaging (MRI). It evaluates AI models' understanding of MRI physics, protocols, image acquisition, and clinical applications.

## Task Description

- **Task Type**: Medical Imaging Knowledge Multiple-Choice QA
- **Input**: MRI-related question with multiple answer choices
- **Output**: Correct answer letter
- **Domain**: Medical imaging, MRI physics, radiology

## Key Features

- Specialized focus on MRI technology and applications
- Tests understanding of MRI physics and protocols
- Covers clinical MRI applications and sequences
- Designed for evaluating medical imaging AI systems
- Multiple-choice format for standardized evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on test split
- Simple accuracy metric
- No training split available

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mri_mcqa` |
| **Dataset ID** | [extraordinarylab/mri-mcqa](https://modelscope.cn/datasets/extraordinarylab/mri-mcqa/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `Medical` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 563 |
| Prompt Length (Mean) | 457.23 chars |
| Prompt Length (Min/Max) | 259 / 888 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "84f179d7",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich cardiac chambers are typically imaged on the short-axis view?\n\nA) RA and RV\nB) RA and LA\nC) LA and LV\nD) RV and LV"
    }
  ],
  "choices": [
    "RA and RV",
    "RA and LA",
    "LA and LV",
    "RV and LV"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets mri_mcqa \
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
    datasets=['mri_mcqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


