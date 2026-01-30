# MMMU-PRO


## Overview

MMMU-PRO is an enhanced multimodal benchmark designed to rigorously assess the genuine understanding capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU benchmark with key improvements that make evaluation more challenging and realistic.

## Task Description

- **Task Type**: Multimodal Academic Question Answering
- **Input**: Images (up to 7) + multiple-choice question
- **Output**: Correct answer choice letter
- **Domains**: 30 academic subjects across STEM, humanities, and social sciences

## Key Features

- Enhanced version of MMMU with more rigorous evaluation
- Covers 30 subjects: Accounting, Biology, Chemistry, Computer Science, Economics, Physics, etc.
- Multiple dataset formats available:
  - `standard (4 options)`: Traditional 4-choice format
  - `standard (10 options)`: Extended 10-choice format for harder evaluation
  - `vision`: Questions embedded in images
- Tests genuine multimodal understanding, not just text shortcuts

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Dataset format can be configured via `dataset_format` parameter
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Rich metadata includes topic difficulty and subject information


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mmmu_pro` |
| **Dataset ID** | [AI-ModelScope/MMMU_Pro](https://modelscope.cn/datasets/AI-ModelScope/MMMU_Pro/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,730 |
| Prompt Length (Mean) | 521.89 chars |
| Prompt Length (Min/Max) | 249 / 3749 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Accounting` | 58 | 518.48 | 320 | 899 |
| `Agriculture` | 60 | 477.05 | 289 | 747 |
| `Architecture_and_Engineering` | 60 | 592.85 | 281 | 1177 |
| `Art` | 53 | 358.34 | 297 | 919 |
| `Art_Theory` | 55 | 362.53 | 289 | 619 |
| `Basic_Medical_Science` | 52 | 401.96 | 277 | 867 |
| `Biology` | 59 | 476.81 | 269 | 1387 |
| `Chemistry` | 60 | 453.75 | 264 | 1217 |
| `Clinical_Medicine` | 59 | 525.58 | 311 | 977 |
| `Computer_Science` | 60 | 441.37 | 262 | 1077 |
| `Design` | 60 | 408.25 | 285 | 1449 |
| `Diagnostics_and_Laboratory_Medicine` | 60 | 444.17 | 274 | 789 |
| `Economics` | 59 | 506.37 | 284 | 900 |
| `Electronics` | 60 | 455.6 | 314 | 668 |
| `Energy_and_Power` | 58 | 506.86 | 347 | 816 |
| `Finance` | 60 | 637.75 | 317 | 1864 |
| `Geography` | 52 | 409.9 | 267 | 929 |
| `History` | 56 | 611.3 | 328 | 1077 |
| `Literature` | 52 | 429.87 | 274 | 564 |
| `Manage` | 50 | 666.56 | 282 | 2198 |
| `Marketing` | 59 | 596.53 | 303 | 1060 |
| `Materials` | 60 | 484.02 | 296 | 1351 |
| `Math` | 60 | 511.95 | 249 | 1172 |
| `Mechanical_Engineering` | 59 | 527.95 | 272 | 1418 |
| `Music` | 60 | 336.55 | 250 | 672 |
| `Pharmacy` | 57 | 474.7 | 282 | 902 |
| `Physics` | 60 | 499.07 | 341 | 737 |
| `Psychology` | 60 | 1355.12 | 280 | 3749 |
| `Public_Health` | 58 | 716.98 | 282 | 2510 |
| `Sociology` | 54 | 416.48 | 279 | 708 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,048 |
| Images per Sample | min: 1, max: 35, mean: 1.18 |
| Resolution Range | 43x50 - 2560x2545 |
| Formats | png |


## Sample Example

**Subset**: `Accounting`

```json
{
  "input": [
    {
      "id": "bae49033",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C. Think step by step before answering.\n\nPrices of zero-coupon bonds reveal the following pattern of forward rates: "
        },
        {
          "image": "[BASE64_IMAGE: png, ~8.0KB]"
        },
        {
          "text": " In addition to the zero-coupon bond, investors also may purchase a 3-year bond making annual payments of $60 with par value $1,000. Under the expectations hypothesis, what is the expected realized compound yield of the coupon bond?\n\nA) 6.66%\nB) 6.79%\nC) 6.91%"
        }
      ]
    }
  ],
  "choices": [
    "6.66%",
    "6.79%",
    "6.91%"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "Accounting",
  "metadata": {
    "id": "test_Accounting_42",
    "explanation": "?",
    "img_type": "['Tables']",
    "topic_difficulty": "Hard",
    "subject": "Accounting"
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

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_format` | `str` | `standard (4 options)` | Dataset format variant. Choices: ['standard (4 options)', 'standard (10 options)', 'vision']. Choices: ['standard (4 options)', 'standard (10 options)', 'vision'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmmu_pro \
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
    datasets=['mmmu_pro'],
    dataset_args={
        'mmmu_pro': {
            # subset_list: ['Accounting', 'Agriculture', 'Architecture_and_Engineering']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


