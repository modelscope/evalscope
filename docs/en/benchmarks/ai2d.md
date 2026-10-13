# AI2D


## Overview

AI2D (AI2 Diagrams) is a benchmark dataset for evaluating AI systems' ability to understand and reason about scientific diagrams. It contains over 5,000 diverse diagrams from science textbooks covering topics like the water cycle, food webs, and biological processes.

## Task Description

- **Task Type**: Diagram Understanding and Visual Reasoning
- **Input**: Scientific diagram image + multiple-choice question
- **Output**: Correct answer choice
- **Domains**: Science education, visual reasoning, diagram comprehension

## Key Features

- Diagrams sourced from real science textbooks
- Requires joint understanding of visual layouts, symbols, and text labels
- Tests interpretation of relationships between diagram elements
- Multiple-choice format with challenging distractors
- Covers diverse scientific domains (biology, physics, earth science)

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Requires understanding both textual labels and visual elements


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ai2d` |
| **Dataset ID** | [evalscope/ai2d](https://modelscope.cn/datasets/evalscope/ai2d/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,088 |
| Prompt Length (Mean) | 324.64 chars |
| Prompt Length (Min/Max) | 256 / 1024 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 3,088 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 177x131 - 1500x1500 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "ce4b20b6",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhat is between the head and abdomen?\n\nA) Antenna\nB) Simple eye\nC) Spiracle\nD) Thorax"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~59.4KB]"
        }
      ]
    }
  ],
  "choices": [
    "Antenna",
    "Simple eye",
    "Spiracle",
    "Thorax"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0
}
```

## Prompt Template

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
    --datasets ai2d \
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
    datasets=['ai2d'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
