# MMLU-Pro


## Overview

MMLU-Pro is an enhanced version of MMLU with increased difficulty and reasoning requirements. It features 10 answer choices instead of 4 and includes more challenging questions requiring deeper reasoning across 14 diverse domains.

## Task Description

- **Task Type**: Multiple-Choice Question Answering (10 options)
- **Input**: Question with up to 10 answer choices
- **Output**: Single correct answer letter (A-J)
- **Domains**: 14 subjects including science, law, engineering, psychology

## Key Features

- 10 answer choices per question (vs 4 in original MMLU)
- More challenging questions requiring reasoning
- Includes Chain-of-Thought (CoT) explanations
- 14 diverse subjects covered
- Reduced impact of random guessing (10% vs 25%)

## Evaluation Notes

- Default configuration uses **5-shot** examples
- Answers should follow "ANSWER: [LETTER]" format
- Uses subject-specific few-shot examples
- Step-by-step reasoning encouraged
- Evaluates on test split with validation for few-shot


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mmlu_pro` |
| **Dataset ID** | [TIGER-Lab/MMLU-Pro](https://modelscope.cn/datasets/TIGER-Lab/MMLU-Pro/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 12,032 |
| Prompt Length (Mean) | 4959.66 chars |
| Prompt Length (Min/Max) | 3048 / 12100 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `computer science` | 410 | 5184.05 | 4696 | 7303 |
| `math` | 1,351 | 4853.54 | 4574 | 7519 |
| `chemistry` | 1,132 | 4567.2 | 4182 | 5975 |
| `engineering` | 969 | 3595.66 | 3048 | 5775 |
| `law` | 1,101 | 7001.35 | 5581 | 9200 |
| `biology` | 717 | 5881.08 | 5179 | 7850 |
| `health` | 818 | 4624.44 | 4134 | 9310 |
| `physics` | 1,299 | 4389.59 | 4004 | 6511 |
| `business` | 789 | 4646.01 | 4316 | 6915 |
| `philosophy` | 499 | 3760.1 | 3360 | 5289 |
| `economics` | 844 | 4680.21 | 4118 | 6907 |
| `other` | 924 | 3730.86 | 3316 | 6108 |
| `psychology` | 798 | 5274.19 | 4656 | 6942 |
| `history` | 381 | 9919.94 | 8711 | 12100 |

## Sample Example

**Subset**: `computer science`

```json
{
  "input": [
    {
      "id": "79336f87",
      "content": "The following are multiple choice questions (with answers) about computer science. Think step by step and then finish your answer with 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the correct letter choice.\n\nQuestion:\nA certain pipel ... [TRUNCATED] ...  random index of a larger value.\nG) The method should be written to return the largest index among larger values.\nH) The method should be written on the assumption that there is only one value in the array that is larger than the given item.\n"
    }
  ],
  "choices": [
    "The method should return an error if more than one larger value is found.",
    "The specification should be modified to indicate what should be done if there is more than one index of larger values.",
    "The method should be written to output a message if more than one larger value is found.",
    "The method should be written so as to return the index of every occurrence of a larger value.",
    "The method should be written to return the last occurrence of a larger value.",
    "The method should return a random index of a larger value.",
    "The method should be written to return the largest index among larger values.",
    "The method should be written on the assumption that there is only one value in the array that is larger than the given item."
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "computer science",
  "metadata": {
    "cot_content": "",
    "subject": "computer science",
    "question_id": 10356
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}

```

<details>
<summary>Few-shot Template</summary>

```text
The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the correct letter choice.

{examples}
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}

```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmlu_pro \
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
    datasets=['mmlu_pro'],
    dataset_args={
        'mmlu_pro': {
            # subset_list: ['computer science', 'math', 'chemistry']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


