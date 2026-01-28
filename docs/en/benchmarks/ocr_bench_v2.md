# OCRBench-v2


## Overview

OCRBench v2 is a large-scale bilingual text-centric benchmark with the most comprehensive set of OCR tasks (4x more than OCRBench v1), covering 31 diverse scenarios including street scenes, receipts, formulas, diagrams, and more.

## Task Description

- **Task Type**: Optical Character Recognition and Document Understanding
- **Input**: Image + OCR/document question
- **Output**: Text recognition, extraction, or analysis result
- **Languages**: English and Chinese (bilingual)

## Key Features

- 10,000 human-verified question-answering pairs
- 31 diverse scenarios (street scene, receipt, formula, diagram, etc.)
- High proportion of difficult samples
- Comprehensive OCR task coverage
- Bilingual (English and Chinese) evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on test split
- Requires: apted, distance, Levenshtein, lxml, Polygon3, zss packages
- Simple accuracy metric


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ocr_bench_v2` |
| **Dataset ID** | [evalscope/OCRBench_v2](https://modelscope.cn/datasets/evalscope/OCRBench_v2/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 10,000 |
| Prompt Length (Mean) | 155.62 chars |
| Prompt Length (Min/Max) | 6 / 1863 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `APP agent en` | 300 | 36.9 | 16 | 84 |
| `ASCII art classification en` | 200 | 174.62 | 162 | 193 |
| `key information extraction cn` | 400 | 32.45 | 9 | 162 |
| `key information extraction en` | 400 | 506.23 | 327 | 1261 |
| `key information mapping en` | 300 | 664.02 | 429 | 1647 |
| `VQA with position en` | 300 | 557.69 | 539 | 612 |
| `chart parsing en` | 400 | 67 | 67 | 67 |
| `cognition VQA cn` | 200 | 15.91 | 6 | 48 |
| `cognition VQA en` | 800 | 44.73 | 14 | 179 |
| `diagram QA en` | 300 | 118.23 | 54 | 356 |
| `document classification en` | 200 | 314 | 314 | 314 |
| `document parsing cn` | 300 | 43 | 43 | 43 |
| `document parsing en` | 400 | 51 | 51 | 51 |
| `formula recognition cn` | 200 | 19 | 19 | 19 |
| `formula recognition en` | 400 | 58.39 | 53 | 60 |
| `handwritten answer extraction cn` | 200 | 39.75 | 22 | 60 |
| `math QA en` | 300 | 119 | 119 | 119 |
| `full-page OCR cn` | 200 | 31 | 31 | 31 |
| `full-page OCR en` | 200 | 91 | 91 | 91 |
| `reasoning VQA en` | 600 | 80.45 | 26 | 256 |
| `reasoning VQA cn` | 400 | 163.96 | 62 | 633 |
| `fine-grained text recognition en` | 200 | 155.63 | 152 | 156 |
| `science QA en` | 300 | 387.54 | 159 | 1863 |
| `table parsing cn` | 300 | 139.4 | 79 | 211 |
| `table parsing en` | 400 | 65.39 | 57 | 134 |
| `text counting en` | 200 | 113.87 | 101 | 127 |
| `text grounding en` | 200 | 361.5 | 357 | 379 |
| `text recognition en` | 800 | 37.26 | 29 | 104 |
| `text spotting en` | 200 | 446 | 446 | 446 |
| `text translation cn` | 400 | 230.88 | 96 | 291 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 10,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 19x10 - 3912x21253 |
| Formats | jpeg |


## Sample Example

**Subset**: `APP agent en`

```json
{
  "input": [
    {
      "id": "51292e32",
      "content": [
        {
          "text": "What is the wrong answer 2?"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~121.7KB]"
        }
      ]
    }
  ],
  "target": "[\"enabled\", \"on\"]",
  "id": 0,
  "group_id": 0,
  "subset_key": "APP agent en",
  "metadata": {
    "question": "What is the wrong answer 2?",
    "answers": [
      "enabled",
      "on"
    ],
    "eval": "None",
    "dataset_name": "rico",
    "type": "APP agent en",
    "bbox": null,
    "bbox_list": null,
    "content": null
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
    --datasets ocr_bench_v2 \
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
    datasets=['ocr_bench_v2'],
    dataset_args={
        'ocr_bench_v2': {
            # subset_list: ['APP agent en', 'ASCII art classification en', 'key information extraction cn']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


