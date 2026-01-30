# MMStar


## Overview

MMStar is an elite vision-indispensable multimodal benchmark designed to ensure genuine visual dependency in evaluation. Each sample is carefully curated to require actual visual understanding, minimizing data leakage and testing advanced multimodal capabilities.

## Task Description

- **Task Type**: Vision-Dependent Multiple-Choice QA
- **Input**: Image + multiple-choice question requiring visual understanding
- **Output**: Single answer letter (A/B/C/D)
- **Domains**: Perception, reasoning, math, science & technology

## Key Features

- Ensures visual dependency - questions cannot be answered without images
- Minimal data leakage from training corpora
- Tests advanced multimodal reasoning capabilities
- Six categories: coarse perception, fine-grained perception, instance reasoning, logical reasoning, math, science & technology
- High-quality curated samples with verified visual necessity

## Evaluation Notes

- Default evaluation uses the **val** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting with "ANSWER: [LETTER]" format
- Results reported per category and overall


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mm_star` |
| **Dataset ID** | [evalscope/MMStar](https://modelscope.cn/datasets/evalscope/MMStar/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `val` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,500 |
| Prompt Length (Mean) | 390.23 chars |
| Prompt Length (Min/Max) | 272 / 2023 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `coarse perception` | 250 | 350.8 | 282 | 784 |
| `fine-grained perception` | 250 | 334.98 | 277 | 608 |
| `instance reasoning` | 250 | 379.02 | 273 | 684 |
| `logical reasoning` | 250 | 427.22 | 284 | 2023 |
| `math` | 250 | 467.36 | 292 | 891 |
| `science & technology` | 250 | 381.98 | 272 | 1173 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,500 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 114x66 - 3160x2136 |
| Formats | jpeg |


## Sample Example

**Subset**: `coarse perception`

```json
{
  "input": [
    {
      "id": "57e256b8",
      "content": [
        {
          "text": "Answer the following multiple choice question.\nThe last line of your response should be of the following format:\n'ANSWER: [LETTER]' (without quotes)\nwhere [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhich option describe the object relationship in the image correctly?\nOptions: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~37.2KB]"
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
  "subset_key": "coarse perception",
  "metadata": {
    "index": 0,
    "category": "coarse perception",
    "l2_category": "image scene and topic",
    "source": "MMBench",
    "split": "val",
    "image_path": "images/0.jpg"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question.
The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes)
where [LETTER] is one of A,B,C,D. Think step by step before answering.

{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mm_star \
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
    datasets=['mm_star'],
    dataset_args={
        'mm_star': {
            # subset_list: ['coarse perception', 'fine-grained perception', 'instance reasoning']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


