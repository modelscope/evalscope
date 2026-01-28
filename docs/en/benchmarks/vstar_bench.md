# V*Bench


## Overview

V*Bench is a benchmark designed for evaluating visual search capabilities within multimodal reasoning systems. It focuses on actively locating and identifying specific visual information in high-resolution images, crucial for fine-grained visual understanding.

## Task Description

- **Task Type**: Visual Search and Reasoning (Multiple-Choice)
- **Input**: High-resolution image + targeted visual query
- **Output**: Answer letter (A/B/C/D)
- **Domains**: Visual search, fine-grained recognition, visual grounding

## Key Features

- Tests targeted visual query capabilities
- Focuses on high-resolution image understanding
- Requires finding and reasoning about specific visual elements
- Questions guided by natural language instructions
- Evaluates fine-grained visual understanding in complex scenes

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting with "ANSWER: [LETTER]" format
- Metadata includes category and question ID for analysis


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `vstar_bench` |
| **Dataset ID** | [lmms-lab/vstar-bench](https://modelscope.cn/datasets/lmms-lab/vstar-bench/summary) |
| **Paper** | N/A |
| **Tags** | `Grounding`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 191 |
| Prompt Length (Mean) | 366.07 chars |
| Prompt Length (Min/Max) | 332 / 402 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 191 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 1690x1500 - 5759x1440 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d0b9c304",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.\n\nWhat is the material of the glove?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather\nAnswer with the option's letter from the given choices directly."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~1.2MB]"
        }
      ]
    }
  ],
  "choices": [
    "A",
    "B",
    "C",
    "D"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "category": "direct_attributes",
    "question_id": "0"
  }
}
```

## Prompt Template

**Prompt Template:**
```text

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vstar_bench \
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
    datasets=['vstar_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


