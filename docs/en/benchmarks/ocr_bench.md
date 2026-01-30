# OCRBench


## Overview

OCRBench is a comprehensive evaluation benchmark designed to assess the OCR (Optical Character Recognition) capabilities of Large Multimodal Models. It covers five key OCR-related tasks with 1,000 manually verified question-answer pairs.

## Task Description

- **Task Type**: OCR and Document Understanding
- **Input**: Image with OCR-related question
- **Output**: Text recognition or extraction result
- **Components**: Text Recognition, VQA, Document VQA, Key Info Extraction, Math Expression Recognition

## Key Features

- 1,000 question-answer pairs across 10 categories
- Manually verified and corrected answers
- Categories include: Regular/Irregular/Artistic/Handwriting Text Recognition
- Scene Text-centric VQA and Document-oriented VQA
- Key Information Extraction
- Handwritten Mathematical Expression Recognition (HME100k)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Simple accuracy metric (inclusion-based matching)
- Results broken down by question type/category
- Different matching rules for HME100k (space-insensitive)
- Comprehensive test of OCR capabilities in multimodal models


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ocr_bench` |
| **Dataset ID** | [evalscope/OCRBench](https://modelscope.cn/datasets/evalscope/OCRBench/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 55.78 chars |
| Prompt Length (Min/Max) | 14 / 149 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Regular Text Recognition` | 50 | 29 | 29 | 29 |
| `Irregular Text Recognition` | 50 | 29 | 29 | 29 |
| `Artistic Text Recognition` | 50 | 29 | 29 | 29 |
| `Handwriting Recognition` | 50 | 29 | 29 | 29 |
| `Digit String Recognition` | 50 | 32 | 32 | 32 |
| `Non-Semantic Text Recognition` | 50 | 29 | 29 | 29 |
| `Scene Text-centric VQA` | 200 | 34.6 | 14 | 101 |
| `Doc-oriented VQA` | 200 | 59 | 21 | 136 |
| `Key Information Extraction` | 200 | 101.58 | 86 | 149 |
| `Handwritten Mathematical Expression Recognition` | 100 | 79 | 79 | 79 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 25x16 - 4961x7016 |
| Formats | jpeg |


## Sample Example

**Subset**: `Regular Text Recognition`

```json
{
  "input": [
    {
      "id": "ff23a835",
      "content": [
        {
          "text": "what is written in the image?"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~1.2KB]"
        }
      ]
    }
  ],
  "target": "[\"CENTRE\"]",
  "id": 0,
  "group_id": 0,
  "subset_key": "Regular Text Recognition",
  "metadata": {
    "dataset": "IIIT5K",
    "question_type": "Regular Text Recognition"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ocr_bench \
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
    datasets=['ocr_bench'],
    dataset_args={
        'ocr_bench': {
            # subset_list: ['Regular Text Recognition', 'Irregular Text Recognition', 'Artistic Text Recognition']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


