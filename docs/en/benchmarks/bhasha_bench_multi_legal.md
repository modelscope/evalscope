# BhashaBench-Multi (Legal)


## Overview

BhashaBench-Multi (Legal) is a domain-specific multiple-choice benchmark evaluating LLM knowledge
of Indian law across 22 Indic languages. Each question originates in English and is machine
translated (with LLM-judged translation quality scores) into the target language; this adapter uses
the translated question/choices.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: A Indian law question with 4 answer choices, in one of 22 Indic languages
- **Output**: Correct answer letter
- **Languages**: Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili,
  Malayalam, Manipuri, Marathi, Nepali, Oriya, Punjabi, Sanskrit, Santhali, Sindhi, Tamil, Telugu, Urdu

## Key Features

- ~14,963 questions per language across 22 Indic languages per domain (~330k total per domain)
- Machine-translated from English with LLM-judged translation quality scores
- 22 scheduled languages of India, all in native script; no English split
- Four domains available as separate benchmarks: Ayurveda, Finance, Krishi, Legal

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate specific languages (e.g., `['Hindi', 'Tamil']`), or `limit` to cap
  sample count — each domain is ~14,963 questions per language across 22 languages (~330k total),
  so evaluating every language's full split is a large run
- No English split exists for this dataset


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bhasha_bench_multi_legal` |
| **Dataset ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 536,030 |
| Prompt Length (Mean) | 490.89 chars |
| Prompt Length (Min/Max) | 225 / 6384 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Assamese` | 24,365 | 475.08 | 232 | 2556 |
| `Bengali` | 24,365 | 482.5 | 235 | 2066 |
| `Bodo` | 24,365 | 521.23 | 225 | 4608 |
| `Dogri` | 24,365 | 487.72 | 225 | 4432 |
| `Gujarati` | 24,365 | 463.64 | 232 | 1954 |
| `Hindi` | 24,365 | 489.37 | 232 | 2202 |
| `Kannada` | 24,365 | 475.35 | 232 | 2068 |
| `Kashmiri` | 24,365 | 514.54 | 242 | 5037 |
| `Konkani` | 24,365 | 473.95 | 225 | 4000 |
| `Maithili` | 24,365 | 473.11 | 225 | 4011 |
| `Malayalam` | 24,365 | 511.29 | 236 | 2218 |
| `Manipuri` | 24,365 | 548.36 | 238 | 6384 |
| `Marathi` | 24,365 | 487.86 | 232 | 2113 |
| `Nepali` | 24,365 | 475.87 | 234 | 2058 |
| `Oriya` | 24,365 | 458.86 | 232 | 1936 |
| `Punjabi` | 24,365 | 483.03 | 232 | 2138 |
| `Sanskrit` | 24,365 | 489.0 | 233 | 1979 |
| `Santhali` | 24,365 | 549.75 | 233 | 5074 |
| `Sindhi` | 24,365 | 455.74 | 234 | 1830 |
| `Tamil` | 24,365 | 522.97 | 237 | 2479 |
| `Telugu` | 24,365 | 481.32 | 234 | 1992 |
| `Urdu` | 24,365 | 479.13 | 235 | 2120 |

## Sample Example

**Subset**: `Assamese`

```json
{
  "input": [
    {
      "id": "e631cc6e",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nকোনো আদেশ প্ৰকাশ কৰাৰ পূৰ্বতে কোনো সমস্যা সংশোধন কৰাৰ বা নতুন সমস্যা উত্থাপন কৰাৰ ক্ষমতা আদালতৰ ওচৰত থাকে, আৰু এই ক্ষমতা দিয়া হয় দেৱানী প্রক্রিয়া বিধি, ১৯০৮-ৰ কোনটো ব্যৱস্থাৰ দ্বাৰা?\n\nA) অধ্যায় ১৪, বিধি ১\nB) অধ্যায় ১৪, বিধি ৫\nC) অধ্যায় XIV, বিধি ৬\nD) ধাৰা ১৫১"
    }
  ],
  "choices": [
    "অধ্যায় ১৪, বিধি ১",
    "অধ্যায় ১৪, বিধি ৫",
    "অধ্যায় XIV, বিধি ৬",
    "ধাৰা ১৫১"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "Assamese",
    "topic": "Procedural Law"
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
    --datasets bhasha_bench_multi_legal \
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
    datasets=['bhasha_bench_multi_legal'],
    dataset_args={
        'bhasha_bench_multi_legal': {
            # subset_list: ['Assamese', 'Bengali', 'Bodo']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
