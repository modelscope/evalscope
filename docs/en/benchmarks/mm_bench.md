# MMBench


## Overview

MMBench is a systematically designed benchmark for evaluating vision-language models across 20 fine-grained ability dimensions. It uses a novel CircularEval strategy and provides both English and Chinese versions for cross-lingual evaluation.

## Task Description

- **Task Type**: Visual Multiple-Choice Q&A
- **Input**: Image with question and 2-4 answer options
- **Output**: Single correct answer letter (A, B, C, or D)
- **Languages**: English (en) and Chinese (cn) subsets

## Key Features

- 20 fine-grained ability dimensions defined
- Systematic evaluation pipeline with CircularEval strategy
- Bilingual support (English and Chinese)
- Questions include hints for context
- Tests perception, reasoning, and knowledge abilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting
- Evaluates on dev split
- Two subsets: `cn` (Chinese) and `en` (English)
- Results include category-level breakdown


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mm_bench` |
| **Dataset ID** | [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `dev` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 8,658 |
| Prompt Length (Mean) | 372.3 chars |
| Prompt Length (Min/Max) | 241 / 2395 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `cn` | 4,329 | 312.28 | 241 | 1681 |
| `en` | 4,329 | 432.33 | 256 | 2395 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 8,658 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 106x56 - 512x512 |
| Formats | jpeg |


## Sample Example

**Subset**: `cn`

```json
{
  "input": [
    {
      "id": "00b49734",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B. Think step by step before answering.\n\n下面的文章描述了一个实验。阅读文章，然后按照以下说明进行操作。\n\nMadelyn在雪板的底部涂上了一层薄蜡，然后直接下坡滑行。然后，她去掉了蜡，再次直接下坡滑行。她重复了这个过程四次，每次都交替使用薄蜡或不使用薄蜡滑行。她的朋友Tucker计时每次滑行的时间。Madelyn和Tucker计算了使用薄蜡滑行和不使用薄蜡滑行时直接下坡所需的平均时间。\n图：滑雪板下坡。麦德琳和塔克的实验能最好回答哪个问题？\n\nA) 当麦德琳的雪板上有一层薄蜡或一层厚蜡时，它是否能在较短的时间内滑下山坡？\nB) 当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~11.7KB]"
        }
      ]
    }
  ],
  "choices": [
    "当麦德琳的雪板上有一层薄蜡或一层厚蜡时，它是否能在较短的时间内滑下山坡？",
    "当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 241,
    "category": "identity_reasoning",
    "source": "scienceqa",
    "L2-category": "attribute_reasoning",
    "comment": "nan",
    "split": "dev"
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
    --datasets mm_bench \
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
    datasets=['mm_bench'],
    dataset_args={
        'mm_bench': {
            # subset_list: ['cn', 'en']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


