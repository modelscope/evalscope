# MMMU


## Overview

MMMU (Massive Multi-discipline Multimodal Understanding) is a comprehensive benchmark designed to evaluate multimodal models on expert-level tasks requiring college-level subject knowledge and deliberate reasoning. It covers 30 subjects across 6 core disciplines.

## Task Description

- **Task Type**: Multimodal Question Answering (Multiple-Choice and Open-Ended)
- **Input**: Questions with diverse images (charts, diagrams, maps, tables, etc.)
- **Output**: Answer letter (MC) or free-form text (Open)
- **Disciplines**: Art & Design, Business, Science, Health & Medicine, Humanities, Tech & Engineering

## Key Features

- 11.5K meticulously collected multimodal questions
- From college exams, quizzes, and textbooks
- 30 subjects and 183 subfields covered
- 30 heterogeneous image types (charts, diagrams, music sheets, chemical structures, etc.)
- Tests both perception and expert-level reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Supports both multiple-choice and open-ended question types
- Multiple images per question supported (up to 7)
- For open questions: "ANSWER: [ANSWER]" format expected
- Evaluates on validation split (test set requires submission)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mmmu` |
| **Dataset ID** | [AI-ModelScope/MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 900 |
| Prompt Length (Mean) | 527.84 chars |
| Prompt Length (Min/Max) | 247 / 3011 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Accounting` | 30 | 525.2 | 356 | 899 |
| `Agriculture` | 30 | 472.73 | 321 | 747 |
| `Architecture_and_Engineering` | 30 | 643.33 | 279 | 927 |
| `Art` | 30 | 384.17 | 297 | 1098 |
| `Art_Theory` | 30 | 370.13 | 297 | 588 |
| `Basic_Medical_Science` | 30 | 420.1 | 277 | 1119 |
| `Biology` | 30 | 524.87 | 294 | 1239 |
| `Chemistry` | 30 | 522.27 | 286 | 1220 |
| `Clinical_Medicine` | 30 | 549.37 | 311 | 914 |
| `Computer_Science` | 30 | 512.73 | 273 | 1285 |
| `Design` | 30 | 401.77 | 292 | 613 |
| `Diagnostics_and_Laboratory_Medicine` | 30 | 440.73 | 302 | 741 |
| `Economics` | 30 | 503.63 | 354 | 730 |
| `Electronics` | 30 | 476.83 | 314 | 659 |
| `Energy_and_Power` | 30 | 529.17 | 347 | 814 |
| `Finance` | 30 | 628.67 | 361 | 985 |
| `Geography` | 30 | 460.37 | 299 | 759 |
| `History` | 30 | 590.63 | 360 | 899 |
| `Literature` | 30 | 425.57 | 275 | 541 |
| `Manage` | 30 | 784 | 414 | 2261 |
| `Marketing` | 30 | 530.6 | 303 | 832 |
| `Materials` | 30 | 474.63 | 308 | 667 |
| `Math` | 30 | 500.6 | 247 | 1167 |
| `Mechanical_Engineering` | 30 | 504.73 | 272 | 861 |
| `Music` | 30 | 315.93 | 250 | 455 |
| `Pharmacy` | 30 | 499.4 | 311 | 981 |
| `Physics` | 30 | 525.83 | 332 | 1086 |
| `Psychology` | 30 | 1173.9 | 280 | 3011 |
| `Public_Health` | 30 | 678.17 | 282 | 2684 |
| `Sociology` | 30 | 465.27 | 299 | 1007 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 980 |
| Images per Sample | min: 1, max: 5, mean: 1.09 |
| Resolution Range | 70x67 - 2560x2133 |
| Formats | png |


## Sample Example

**Subset**: `Accounting`

```json
{
  "input": [
    {
      "id": "be505d26",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~65.8KB]"
        },
        {
          "text": " Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?\n\nA) $6\nB) $7\nC) $8\nD) $9"
        }
      ]
    }
  ],
  "choices": [
    "$6",
    "$7",
    "$8",
    "$9"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "validation_Accounting_1",
    "question_type": "multiple-choice",
    "subfield": "Managerial Accounting",
    "explanation": "",
    "img_type": "['Tables']",
    "topic_difficulty": "Medium"
  }
}
```

## Prompt Template

**Prompt Template:**
```text

Solve the following problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \boxed command.


```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmmu \
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
    datasets=['mmmu'],
    dataset_args={
        'mmmu': {
            # subset_list: ['Accounting', 'Agriculture', 'Architecture_and_Engineering']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


