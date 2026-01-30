# BioMixQA

## Overview

BiomixQA is a curated biomedical question-answering dataset designed to evaluate AI models on biomedical knowledge and reasoning. It has been utilized to validate the Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) framework across different LLMs.

## Task Description

- **Task Type**: Biomedical Multiple-Choice Question Answering
- **Input**: Biomedical question with multiple answer choices
- **Output**: Correct answer letter
- **Domain**: Biomedical sciences, healthcare, life sciences

## Key Features

- Curated biomedical questions from diverse sources
- Tests medical and biological knowledge comprehension
- Validates RAG framework effectiveness for biomedical domain
- Multiple-choice format for standardized evaluation
- Useful for evaluating healthcare AI systems

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Simple accuracy metric for performance measurement
- Evaluates on test split
- No few-shot examples provided

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `biomix_qa` |
| **Dataset ID** | [extraordinarylab/biomix-qa](https://modelscope.cn/datasets/extraordinarylab/biomix-qa/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `Medical` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 306 |
| Prompt Length (Mean) | 344.93 chars |
| Prompt Length (Min/Max) | 316 / 393 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "5e830918",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E.\n\nOut of the given list, which Gene is associated with head and neck cancer and uveal melanoma.\n\nA) ABO\nB) CACNA2D1\nC) PSCA\nD) TERT\nE) SULT1B1"
    }
  ],
  "choices": [
    "ABO",
    "CACNA2D1",
    "PSCA",
    "TERT",
    "SULT1B1"
  ],
  "target": "B",
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
    --datasets biomix_qa \
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
    datasets=['biomix_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


