# AGIEval


## Overview

AGIEval is a human-centric benchmark designed to evaluate foundation models in the context of human cognition and problem-solving. It uses official, standard, and authoritative admission and qualification exams intended for general human test-takers, such as college entrance exams (GaoKao), law school admission tests (LSAT), math competitions, and lawyer qualification exams.

## Task Description

- **Task Type**: Mixed (Multiple-Choice QA + Open-ended Math)
- **Input**: Questions from standardized exams with optional passages and answer choices
- **Output**: Answer letter(s) for MCQ, or numerical/mathematical answer for open-ended
- **Languages**: English and Chinese

## Key Features

- 21 subsets covering diverse exam types across two languages
- English MCQ: LSAT (AR/LR/RC), SAT (Math/English), AQuA-RAT, LogiQA, GaoKao-English
- Chinese MCQ: GaoKao (Chinese/Geography/History/Biology/Chemistry/Physics/MathQA), LogiQA-zh, JEC-QA
- Open-ended math: MATH (English), GaoKao-MathCloze (Chinese)
- Multi-select subsets: JEC-QA-KD, JEC-QA-CA, GaoKao-Physics
- Includes passage-based reading comprehension questions

## Evaluation Notes

- MCQ subsets use evalscope's standard MultiChoice template and extraction
- Multi-select subsets use Chinese multi-answer template
- Math/cloze subsets use mathematical equivalence checking
- CoT (Chain-of-Thought) prompting enabled by default


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `agieval` |
| **Dataset ID** | [opencompass/agieval](https://modelscope.cn/datasets/opencompass/agieval/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `dev` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 8,269 |
| Prompt Length (Mean) | 673.58 chars |
| Prompt Length (Min/Max) | 40 / 5316 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `aqua-rat` | 254 | 290.09 | 103 | 587 |
| `logiqa-en` | 651 | 911.89 | 248 | 1769 |
| `lsat-ar` | 230 | 946.36 | 635 | 1853 |
| `lsat-lr` | 510 | 1156.66 | 563 | 2348 |
| `lsat-rc` | 269 | 3652.86 | 2959 | 4825 |
| `sat-math` | 220 | 392.45 | 120 | 1201 |
| `sat-en` | 206 | 4618.28 | 3569 | 5316 |
| `sat-en-without-passage` | 206 | 435.91 | 169 | 937 |
| `gaokao-english` | 306 | 2025.44 | 517 | 4216 |
| `logiqa-zh` | 651 | 267.62 | 98 | 526 |
| `gaokao-chinese` | 246 | 988.09 | 152 | 2186 |
| `gaokao-geography` | 199 | 204.82 | 64 | 881 |
| `gaokao-history` | 235 | 141.48 | 67 | 314 |
| `gaokao-biology` | 210 | 203.98 | 75 | 685 |
| `gaokao-chemistry` | 207 | 348.37 | 58 | 1454 |
| `gaokao-physics` | 200 | 251.9 | 58 | 581 |
| `gaokao-mathqa` | 351 | 201.59 | 93 | 615 |
| `jec-qa-kd` | 1,000 | 170.43 | 54 | 454 |
| `jec-qa-ca` | 1,000 | 240.71 | 79 | 883 |
| `math` | 1,000 | 211.95 | 40 | 2186 |
| `gaokao-mathcloze` | 118 | 123.42 | 48 | 501 |

## Sample Example

**Subset**: `aqua-rat`

```json
{
  "input": [
    {
      "id": "e28353e5",
      "content": "Q: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower? Answer Choices: (A)5(√3 + 1) (B)6(√3 + √2) (C)7(√3 – 1) (D)8(√3 – 2) (E)None of these\nA: Among A through E, the answer is"
    }
  ],
  "choices": [
    "(A)5(√3 + 1)",
    "(B)6(√3 + √2)",
    "(C)7(√3 – 1)",
    "(D)8(√3 – 2)",
    "(E)None of these"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "subset": "aqua-rat",
    "has_passage": false
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
    --datasets agieval \
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
    datasets=['agieval'],
    dataset_args={
        'agieval': {
            # subset_list: ['aqua-rat', 'logiqa-en', 'lsat-ar']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


