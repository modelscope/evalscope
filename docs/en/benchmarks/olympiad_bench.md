# OlympiadBench


## Overview

OlympiadBench is an Olympiad-level bilingual multimodal scientific benchmark featuring 8,476 problems from mathematics and physics competitions, including the Chinese college entrance exam (CEE). It provides rigorous evaluation of advanced scientific reasoning.

## Task Description

- **Task Type**: Olympiad-Level Math/Physics Problem Solving
- **Input**: Problem text with optional images (up to 9)
- **Output**: Mathematical answer or proof
- **Domains**: Mathematics, Physics (bilingual: English and Chinese)

## Key Features

- 8,476 Olympiad-level problems
- Bilingual support (English and Chinese)
- Covers both Mathematics and Physics
- Subset naming convention:
  - `OE`: Open-Ended problems
  - `TP`: Theorem Proving problems
  - `MM`: Multimodal (with images)
  - `TO`: Text-Only
  - `CEE`: Chinese Entrance Exam
  - `COMP`: Comprehensive competition problems

## Evaluation Notes

- Default evaluation uses the **train** split
- Primary metric: **Accuracy** with mathematical judging
- Answers should be in \boxed{} format
- **Note**: `TP` (Theorem Proving) subsets cannot be auto-evaluated currently
- Supports numerical precision/error thresholds for approximate answers


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `olympiad_bench` |
| **Dataset ID** | [AI-ModelScope/OlympiadBench](https://modelscope.cn/datasets/AI-ModelScope/OlympiadBench/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 8,476 |
| Prompt Length (Mean) | 484.46 chars |
| Prompt Length (Min/Max) | 129 / 5032 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `OE_MM_maths_en_COMP` | 150 | 1118.59 | 575 | 3943 |
| `OE_MM_maths_zh_CEE` | 1,910 | 343.89 | 182 | 1566 |
| `OE_MM_maths_zh_COMP` | 56 | 379.07 | 201 | 608 |
| `OE_MM_physics_en_COMP` | 456 | 863.57 | 556 | 2793 |
| `OE_MM_physics_zh_CEE` | 1,483 | 425.68 | 214 | 901 |
| `OE_TO_maths_en_COMP` | 674 | 825.93 | 545 | 4924 |
| `OE_TO_maths_zh_CEE` | 1,240 | 297.26 | 171 | 1098 |
| `OE_TO_maths_zh_COMP` | 408 | 322.72 | 173 | 1583 |
| `OE_TO_physics_en_COMP` | 236 | 949.1 | 548 | 3112 |
| `OE_TO_physics_zh_CEE` | 115 | 329.56 | 203 | 503 |
| `TP_MM_maths_en_COMP` | 62 | 2044.19 | 475 | 4617 |
| `TP_MM_maths_zh_CEE` | 652 | 241.05 | 165 | 674 |
| `TP_MM_maths_zh_COMP` | 81 | 304.7 | 206 | 512 |
| `TP_MM_physics_en_COMP` | 19 | 677.79 | 432 | 1602 |
| `TP_TO_maths_en_COMP` | 503 | 933.44 | 449 | 5032 |
| `TP_TO_maths_zh_CEE` | 207 | 242.79 | 141 | 947 |
| `TP_TO_maths_zh_COMP` | 199 | 285.43 | 129 | 522 |
| `TP_TO_physics_en_COMP` | 25 | 740.48 | 475 | 1262 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 5,875 |
| Images per Sample | min: 1, max: 9, mean: 1.21 |
| Resolution Range | 64x46 - 1765x1947 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `OE_MM_maths_en_COMP`

```json
{
  "input": [
    {
      "id": "d7b44a52",
      "content": [
        {
          "text": "The following is an open-ended problem from an International Math competition. The answer of The problem should be a numerical value. Please calculate the answer according to the given requirements and the information provided. Please use LaT ... [TRUNCATED] ... c_{3}, \\ldots$ with $c_{i}<C$ for all $i$, Turbo can (after studying the sequence) ensure that there is some point on the circle that it will never visit or crawl across.\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        },
        {
          "image": "[BASE64_IMAGE: jpg, ~27.4KB]"
        }
      ]
    }
  ],
  "target": "$\\frac{1}{2}$",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": 2231,
    "subfield": "Geometry",
    "context": null,
    "solution": [
      "The largest possible $C$ is $C=\\frac{1}{2}$.\n\nFor $0<C \\leqslant \\frac{1}{2}$, Turbo can simply choose an arbitrary point $P$ (different from its starting point) to avoid. When Turbo is at an arbitrary point $A$ different from $P$, the two ar ... [TRUNCATED] ... Note: Every sequence of the form $c_{i}=x$ if $i$ is odd, and $c_{i}=y$ if $i$ is even, where $0<x, y<C$, such that $x+y \\geqslant 1$, and $x \\neq y$ satisfies the conditions with the same argument. There might be even more possible examples.",
      "To show that $C\\le \\frac12$\n\nWe consider the following related problem:\n\nWe assume instead that the snail Chet is moving left and right on the real line. Find the size $M$ of the smallest (closed) interval, that we cannot force Chet out of, u ... [TRUNCATED] ... n, 1-\\varepsilon]$. Indeed the absolute value of the final position is at least $1-\\frac{5}{6} \\varepsilon$. This contradicts the assumption, that we cannot force Chet out of $[-1+\\varepsilon, 1-\\varepsilon]$. Hence $M \\geqslant 2$ as needed."
    ],
    "final_answer": [
      "$\\frac{1}{2}$"
    ],
    "is_multiple_answer": false,
    "unit": null,
    "answer_type": "Numerical",
    "question_type": "Open-ended",
    "language": "English",
    "subject": "Math",
    "error": null
  }
}
```

*Note: Some content was truncated for display.*

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
    --datasets olympiad_bench \
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
    datasets=['olympiad_bench'],
    dataset_args={
        'olympiad_bench': {
            # subset_list: ['OE_MM_maths_en_COMP', 'OE_MM_maths_zh_CEE', 'OE_MM_maths_zh_COMP']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


