# ARC


## Overview

ARC (AI2 Reasoning Challenge) is a benchmark designed to evaluate science question answering capabilities of AI models. It consists of multiple-choice science questions from grade 3 to grade 9, divided into an Easy set and a Challenge set based on difficulty.

## Task Description

- **Task Type**: Multiple-Choice Science Question Answering
- **Input**: Science question with 3-5 answer choices
- **Output**: Correct answer letter (A, B, C, D, or E)
- **Difficulty Levels**: ARC-Easy and ARC-Challenge

## Key Features

- 7,787 science questions from standardized tests (grades 3-9)
- ARC-Easy: Questions answerable by retrieval or word co-occurrence
- ARC-Challenge: Questions requiring deeper reasoning
- Questions cover physics, chemistry, biology, and earth science
- Designed to test both factual knowledge and reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Two subsets available: `ARC-Easy` and `ARC-Challenge`
- Challenge set is commonly used for leaderboard comparisons
- Supports few-shot evaluation with train split examples


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `arc` |
| **Dataset ID** | [allenai/ai2_arc](https://modelscope.cn/datasets/allenai/ai2_arc/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,548 |
| Prompt Length (Mean) | 424.43 chars |
| Prompt Length (Min/Max) | 253 / 1157 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `ARC-Easy` | 2,376 | 409.9 | 253 | 1157 |
| `ARC-Challenge` | 1,172 | 453.91 | 253 | 1111 |

## Sample Example

**Subset**: `ARC-Easy`

```json
{
  "input": [
    {
      "id": "76edd7a5",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich statement best explains why photosynthesis is the foundation of most food webs?\n\nA) Sunlight is the source of energy for nearly all ecosystems.\nB) Most ecosystems are found on land instead of in water.\nC) Carbon dioxide is more available than other gases.\nD) The producers in all ecosystems are plants."
    }
  ],
  "choices": [
    "Sunlight is the source of energy for nearly all ecosystems.",
    "Most ecosystems are found on land instead of in water.",
    "Carbon dioxide is more available than other gases.",
    "The producers in all ecosystems are plants."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "Mercury_417466"
  }
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
    --datasets arc \
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
    datasets=['arc'],
    dataset_args={
        'arc': {
            # subset_list: ['ARC-Easy', 'ARC-Challenge']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


