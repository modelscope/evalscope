# SEED-Bench-2-Plus


## Overview

SEED-Bench-2-Plus is a large-scale benchmark designed to evaluate Multimodal Large Language Models (MLLMs) on text-rich visual understanding tasks. It contains 2.3K multiple-choice questions with precise human annotations across real-world scenarios.

## Task Description

- **Task Type**: Text-Rich Visual Question Answering
- **Input**: Image containing text-rich content + multiple-choice question
- **Output**: Correct answer choice letter (A/B/C/D)
- **Domains**: Charts, maps, web interfaces

## Key Features

- Focuses on text-rich visual scenarios common in real applications
- Three broad categories: **Charts**, **Maps**, and **Webs**
- Human-annotated questions with high quality
- Tests understanding of complex visual layouts with text
- Multiple difficulty levels within each category

## Evaluation Notes

- Default evaluation uses the **test** split
- Available subsets: `chart`, `web`, `map`
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Rich metadata including data source, type, and difficulty level


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `seed_bench_2_plus` |
| **Dataset ID** | [evalscope/SEED-Bench-2-Plus](https://modelscope.cn/datasets/evalscope/SEED-Bench-2-Plus/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,277 |
| Prompt Length (Mean) | 393.01 chars |
| Prompt Length (Min/Max) | 280 / 710 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `chart` | 810 | 390.92 | 280 | 663 |
| `web` | 660 | 395.56 | 294 | 664 |
| `map` | 807 | 393.02 | 282 | 710 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,277 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 800x800 - 800x800 |
| Formats | png |


## Sample Example

**Subset**: `chart`

```json
{
  "input": [
    {
      "id": "7fdf638f",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nAccording to the tree diagram, how many women who have breast cancer received a negative mammogram?\n\nA) 80\nB) 950\nC) 20\nD) 8,950"
        },
        {
          "image": "[BASE64_IMAGE: png, ~68.1KB]"
        }
      ]
    }
  ],
  "choices": [
    "80",
    "950",
    "20",
    "8,950"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "chart",
  "metadata": {
    "data_id": "text_rich/1.png",
    "question_id": "0",
    "question_image_subtype": "tree diagram",
    "data_source": "SEED-Bench v2 plus",
    "data_type": "Single Image",
    "level": "L1",
    "subpart": "Single-Image & Text Comprehension",
    "version": "v2+"
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
    --datasets seed_bench_2_plus \
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
    datasets=['seed_bench_2_plus'],
    dataset_args={
        'seed_bench_2_plus': {
            # subset_list: ['chart', 'web', 'map']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


