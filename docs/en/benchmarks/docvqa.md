# DocVQA


## Overview

DocVQA (Document Visual Question Answering) is a benchmark designed to evaluate AI systems' ability to answer questions based on document images such as scanned pages, forms, invoices, and reports. It requires understanding complex document layouts, structure, and visual elements beyond simple text extraction.

## Task Description

- **Task Type**: Document Visual Question Answering
- **Input**: Document image + natural language question
- **Output**: Single word or phrase answer extracted from document
- **Domains**: Document understanding, OCR, layout comprehension

## Key Features

- Covers diverse document types (forms, invoices, letters, reports)
- Requires understanding document layout and structure
- Tests both text extraction and contextual reasoning
- Questions require locating and interpreting specific information
- Combines OCR capabilities with visual understanding

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **ANLS** (Average Normalized Levenshtein Similarity)
- Answers should be in format "ANSWER: [ANSWER]"
- ANLS metric accounts for minor OCR/spelling variations
- Multiple valid answers may be accepted for each question


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `docvqa` |
| **Dataset ID** | [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `anls` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 5,349 |
| Prompt Length (Mean) | 254.82 chars |
| Prompt Length (Min/Max) | 220 / 354 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 5,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 593x294 - 5367x7184 |
| Formats | png |


## Sample Example

**Subset**: `DocVQA`

```json
{
  "input": [
    {
      "id": "002390bd",
      "content": [
        {
          "text": "Answer the question according to the image using a single word or phrase.\nWhat is the ‘actual’ value per 1000, during the year 1975?\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the question."
        },
        {
          "image": "[BASE64_IMAGE: png, ~1.2MB]"
        }
      ]
    }
  ],
  "target": "[\"0.28\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "questionId": "49153",
    "question_types": [
      "figure/diagram"
    ],
    "docId": 14465,
    "ucsf_document_id": "pybv0228",
    "ucsf_document_page_no": "81"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets docvqa \
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
    datasets=['docvqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


