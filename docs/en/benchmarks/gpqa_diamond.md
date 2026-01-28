# GPQA-Diamond


## Overview

GPQA (Graduate-Level Google-Proof Q&A) Diamond is a challenging benchmark of 198 multiple-choice questions written by domain experts in biology, physics, and chemistry. The questions are designed to be extremely difficult, requiring PhD-level expertise to answer correctly.

## Task Description

- **Task Type**: Expert-Level Multiple-Choice Q&A
- **Input**: Graduate-level science question with 4 choices
- **Output**: Single correct answer letter (A, B, C, or D)
- **Domains**: Biology, Physics, Chemistry

## Key Features

- 198 questions written and validated by domain PhD experts
- Questions are "Google-proof" - cannot be easily looked up
- Designed to test deep domain knowledge and reasoning
- Diamond subset represents the highest quality questions
- Average human expert accuracy ~65%, non-expert ~34%

## Evaluation Notes

- Default configuration uses **0-shot** or **5-shot** evaluation
- Supports Chain-of-Thought (CoT) prompting for improved reasoning
- Answer choices are randomly shuffled during evaluation
- Only uses train split (validation set is private)
- Challenging benchmark for measuring expert-level reasoning


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gpqa_diamond` |
| **Dataset ID** | [AI-ModelScope/gpqa_diamond](https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 198 |
| Prompt Length (Mean) | 841.15 chars |
| Prompt Length (Min/Max) | 340 / 5845 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "82b448a9",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nTwo quantum states wi ... [TRUNCATED] ...  and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\n\nA) 10^-4 eV\nB) 10^-9 eV\nC) 10^-8 eV\nD) 10^-11 eV"
    }
  ],
  "choices": [
    "10^-4 eV",
    "10^-9 eV",
    "10^-8 eV",
    "10^-11 eV"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "",
  "metadata": {
    "correct_answer": "10^-4 eV",
    "incorrect_answers": [
      "10^-11 eV",
      "10^-8 eV\n",
      "10^-9 eV"
    ]
  }
}
```

*Note: Some content was truncated for display.*

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
    --datasets gpqa_diamond \
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
    datasets=['gpqa_diamond'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


