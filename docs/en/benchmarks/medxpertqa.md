# MedXpertQA


## Overview

MedXpertQA is an expert-level medical multiple-choice benchmark designed to evaluate advanced
medical knowledge and reasoning. It contains separate text-only and multimodal tracks built from
challenging medical examination questions and reviewed by licensed physicians.

## Task Description

- **Task Type**: Single-answer medical multiple choice
- **Input**: A clinical or biomedical question with answer choices, optionally accompanied by up to six images
- **Output**: One answer letter (A-J for Text or A-E for MM)
- **Domain**: Medicine across 17 specialties and 11 human body systems

## Key Features

- The test split contains 4,450 questions: 2,450 Text questions with ten options and 2,000 MM questions with five options
- The MM track contains radiology, pathology, optical, photographic, diagram, chart, table, document, and vital-sign imagery
- Questions are annotated by medical task, body system, and question type; 3,307 test questions require reasoning and 1,143 assess understanding
- Questions underwent difficulty filtering, option augmentation, leakage mitigation, and multiple rounds of expert review

## Evaluation Notes

- Primary metric: **Accuracy** by exact match of the predicted option letter
- The default prompt uses EvalScope's zero-shot chain-of-thought template, preserving the official step-by-step instruction and exact answer-letter scoring
- Set `max_tokens` high enough for the model to emit the required final `ANSWER: [LETTER]` line; truncated reasoning may otherwise fall back to the shared parser's last valid uppercase letter
- Results are reported separately for the Text and MM subsets and combined with sample-weighted aggregation
- MM images are stored in `images.zip` (about 517 MB) and read directly from the archive without extracting a second copy
- The published dataset has 4,460 records including ten development examples; this integration evaluates the 4,450 held-out test questions
- [Paper](https://arxiv.org/abs/2501.18362) | [GitHub](https://github.com/TsinghuaC3I/MedXpertQA)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `medxpertqa` |
| **Dataset ID** | [evalscope/MedXpertQA](https://modelscope.cn/datasets/evalscope/MedXpertQA/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2501.18362) |
| **Tags** | `MCQ`, `Medical`, `MultiModal`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,450 |
| Prompt Length (Mean) | 1135.22 chars |
| Prompt Length (Min/Max) | 346 / 4771 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Text` | 2,450 | 1337.92 | 435 | 4771 |
| `MM` | 2,000 | 886.91 | 346 | 2335 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,852 |
| Images per Sample | min: 1, max: 6, mean: 1.43 |
| Resolution Range | 323x34 - 4248x2144 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `Text`

```json
{
  "input": [
    {
      "id": "3f1d2f2a",
      "content": "You are a helpful medical assistant."
    },
    {
      "id": "1a9f9143",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J. Think step by step before answering.\n\nWhich pat ... [TRUNCATED 885 chars] ... ere posterior wear undergoing shoulder arthroplasty\nI) 58-year-old male with glenoid retroversion of 12-degrees undergoing shoulder arthroplasty\nJ) 55-year-old male with glenoid retroversion of 8-degrees undergoing total shoulder arthroplasty"
        }
      ]
    }
  ],
  "choices": [
    "70-year-old male with glenoid retroversion of 18-degrees undergoing shoulder arthroplasty",
    "70-year-old female with humeral anteversion of 13-degrees undergoing shoulder arthroplasty",
    "63-year-old female with glenoid retroversion of 22-degrees and mild posterior wear undergoing shoulder arthroplasty",
    "65-year-old female with glenoid retroversion of 25-degrees undergoing shoulder arthroplasty",
    "65-year-old female with a glenoid retroversion of 13-degrees undergoing shoulder arthroplasty",
    "68-year-old female with glenoid retroversion of 20-degrees undergoing reverse shoulder arthroplasty",
    "72-year-old male with glenoid retroversion of 15-degrees undergoing shoulder arthroplasty",
    "65-year-old female with glenoid retroversion of 30-degrees and severe posterior wear undergoing shoulder arthroplasty",
    "58-year-old male with glenoid retroversion of 12-degrees undergoing shoulder arthroplasty",
    "55-year-old male with glenoid retroversion of 8-degrees undergoing total shoulder arthroplasty"
  ],
  "target": "E",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "Text-0",
    "medical_task": "Basic Science",
    "body_system": "Skeletal",
    "question_type": "Reasoning",
    "images": []
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**System Prompt:**
```text
You are a helpful medical assistant.
```

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets medxpertqa \
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
    datasets=['medxpertqa'],
    dataset_args={
        'medxpertqa': {
            # subset_list: ['Text', 'MM']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
