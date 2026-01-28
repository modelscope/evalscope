# A-OKVQA


## Overview

A-OKVQA (Augmented OK-VQA) is a benchmark designed to evaluate commonsense reasoning and external world knowledge in visual question answering. It extends beyond basic VQA tasks that rely solely on image content, requiring models to leverage a broad spectrum of commonsense and factual knowledge about the world.

## Task Description

- **Task Type**: Visual Question Answering with Knowledge Reasoning
- **Input**: Image + natural language question requiring external knowledge
- **Output**: Answer (multiple-choice or open-ended)
- **Domains**: Commonsense reasoning, factual knowledge, visual understanding

## Key Features

- Requires commonsense reasoning beyond direct visual observation
- Combines visual understanding with external world knowledge
- Includes both multiple-choice and open-ended question formats
- Questions annotated with rationales explaining the reasoning process
- More challenging than standard VQA benchmarks

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **Accuracy** for multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for better reasoning
- Questions require reasoning beyond what is directly visible in images


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `a_okvqa` |
| **Dataset ID** | [HuggingFaceM4/A-OKVQA](https://modelscope.cn/datasets/HuggingFaceM4/A-OKVQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,145 |
| Prompt Length (Mean) | 310.84 chars |
| Prompt Length (Min/Max) | 276 / 405 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,145 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 305x229 - 640x640 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "3d2b0351",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhat is in the motorcyclist's mouth?\n\nA) toothpick\nB) food\nC) popsicle stick\nD) cigarette"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~53.4KB]"
        }
      ]
    }
  ],
  "choices": [
    "toothpick",
    "food",
    "popsicle stick",
    "cigarette"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "22jbM6gDxdaMaunuzgrsBB",
    "direct_answers": "['cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette']",
    "difficult_direct_answer": false,
    "rationales": [
      "He's smoking while riding.",
      "The motorcyclist has a lit cigarette in his mouth while he rides on the street.",
      "The man is smoking."
    ]
  }
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
    --datasets a_okvqa \
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
    datasets=['a_okvqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


