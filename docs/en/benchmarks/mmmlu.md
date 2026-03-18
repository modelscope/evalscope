# MMMLU


## Overview

MMMLU (Multilingual Massive Multitask Language Understanding) is a multilingual extension of the MMLU benchmark. It evaluates the multilingual knowledge and reasoning capabilities of language models across 14 languages, covering 57 subjects from the original MMLU benchmark.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: Question with four answer choices (A, B, C, D) in one of 14 languages
- **Output**: Single correct answer letter
- **Languages**: Arabic, Bengali, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Portuguese, Swahili, Yoruba, Chinese
- **Subjects**: 57 subjects from MMLU (STEM, Humanities, Social Sciences, Other)

## Key Features

- Multilingual translation of the full MMLU benchmark
- 14 typologically diverse languages covering major language families
- Tests cross-lingual knowledge transfer and multilingual reasoning
- Same subject coverage as original MMLU (57 subjects)
- Includes low-resource languages (e.g., Swahili, Yoruba)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split only)
- Use `subset_list` to evaluate specific languages (e.g., `['ZH_CN', 'JA_JP', 'FR_FR']`)
- Results are grouped by language subset
- Cross-lingual performance comparison supported


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mmmlu` |
| **Dataset ID** | [openai-mirror/MMMLU](https://modelscope.cn/datasets/openai-mirror/MMMLU/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 196,588 |
| Prompt Length (Mean) | 624.75 chars |
| Prompt Length (Min/Max) | 136 / 5975 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `AR_XY` | 14,042 | 584.94 | 231 | 4735 |
| `BN_BD` | 14,042 | 654.99 | 247 | 4914 |
| `DE_DE` | 14,042 | 791.64 | 294 | 5657 |
| `ES_LA` | 14,042 | 753.18 | 271 | 5791 |
| `FR_FR` | 14,042 | 777.82 | 278 | 5952 |
| `HI_IN` | 14,042 | 675.02 | 256 | 5379 |
| `ID_ID` | 14,042 | 726.51 | 270 | 5539 |
| `IT_IT` | 14,042 | 761.19 | 277 | 5975 |
| `JA_JP` | 14,042 | 322.79 | 149 | 2064 |
| `KO_KR` | 14,042 | 354.35 | 153 | 2345 |
| `PT_BR` | 14,042 | 706.79 | 258 | 5635 |
| `SW_KE` | 14,042 | 699.08 | 259 | 5566 |
| `YO_NG` | 14,042 | 681.01 | 248 | 5644 |
| `ZH_CN` | 14,042 | 257.15 | 136 | 1495 |

## Sample Example

**Subset**: `AR_XY`

```json
{
  "input": [
    {
      "id": "e43faf14",
      "content": "أجب على سؤال الاختيار من متعدد التالي. يجب أن يكون السطر الأخير من إجابتك بالتنسيق التالي: 'ANSWER: [LETTER]' (بدون علامات اقتباس) حيث [LETTER] هو أحد الحروف A,B,C,D. فكّر خطوة بخطوة قبل الإجابة.\n\nأوجد درجة امتداد الحقل المحدد Q(sqrt(2)، sqrt(3)، sqrt(18)) على Q.\n\nA) 0\nB) 4\nC) 2\nD) 6"
    }
  ],
  "choices": [
    "0",
    "4",
    "2",
    "6"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "subject": "abstract_algebra",
    "language": "AR_XY"
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
    --datasets mmmlu \
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
    datasets=['mmmlu'],
    dataset_args={
        'mmmlu': {
            # subset_list: ['AR_XY', 'BN_BD', 'DE_DE']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


