# MathVerse


## Overview

MathVerse is an all-around visual math benchmark designed for equitable and in-depth evaluation of Multimodal Large Language Models (MLLMs). It contains 2,612 high-quality, multi-subject math problems with diagrams, transformed into 15K test samples across varying information modalities.

## Task Description

- **Task Type**: Visual Mathematical Reasoning
- **Input**: Math problem with diagram + question (multi-choice or free-form)
- **Output**: Answer (letter for multi-choice, numerical/expression for free-form)
- **Domains**: Multi-subject mathematics with visual diagrams

## Key Features

- 2,612 problems transformed into 6 versions each (15K total samples)
- Tests whether MLLMs truly understand visual diagrams for math reasoning
- Problem versions vary by visual information dependency:
  - **Text Dominant**: Most info in text
  - **Text Lite**: Balanced text/visual
  - **Vision Intensive**: More visual reliance
  - **Vision Dominant**: Primarily visual
  - **Vision Only**: All info in diagram
- Supports both multiple-choice and free-form answers

## Evaluation Notes

- Default evaluation uses the **testmini** split
- Primary metric: **Accuracy** with numeric comparison
- Free-form answers use \boxed{} format
- Uses LLM judge for answer verification
- Results reported per problem version for detailed analysis


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `math_verse` |
| **Dataset ID** | [evalscope/MathVerse](https://modelscope.cn/datasets/evalscope/MathVerse/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `testmini` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,940 |
| Prompt Length (Mean) | 274.2 chars |
| Prompt Length (Min/Max) | 70 / 1535 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Text Dominant` | 788 | 369.63 | 122 | 1535 |
| `Text Lite` | 788 | 294.77 | 78 | 1397 |
| `Vision Intensive` | 788 | 280.39 | 78 | 1350 |
| `Vision Dominant` | 788 | 272.11 | 78 | 1356 |
| `Vision Only` | 788 | 154.1 | 70 | 222 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 3,940 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 63x70 - 6840x3549 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `Text Dominant`

```json
{
  "input": [
    {
      "id": "a3189330",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.\n\nAs shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°"
        },
        {
          "image": "[BASE64_IMAGE: png, ~1.6KB]"
        }
      ]
    }
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "Text Dominant",
  "metadata": {
    "sample_index": "1",
    "problem_index": "1",
    "problem_version": "Text Dominant",
    "question_type": "multi-choice",
    "query_wo": "Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.\nQuestion: As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°",
    "query_cot": "Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°",
    "question_for_eval": "As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets math_verse \
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
    datasets=['math_verse'],
    dataset_args={
        'math_verse': {
            # subset_list: ['Text Dominant', 'Text Lite', 'Vision Intensive']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


