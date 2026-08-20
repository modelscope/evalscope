# BhashaBench-Multi (Ayurveda)


## Overview

BhashaBench-Multi (Ayurveda) is a domain-specific multiple-choice benchmark evaluating LLM knowledge
of Ayurvedic medicine across 22 Indic languages. Each question originates in English and is machine
translated (with LLM-judged translation quality scores) into the target language; this adapter uses
the translated question/choices.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: A Ayurvedic medicine question with 4 answer choices, in one of 22 Indic languages
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
| **Benchmark Name** | `bhasha_bench_multi_ayur` |
| **Dataset ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 329,186 |
| Prompt Length (Mean) | 317.8 chars |
| Prompt Length (Min/Max) | 220 / 8370 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Assamese` | 14,963 | 325.76 | 229 | 4447 |
| `Bengali` | 14,963 | 313.28 | 231 | 1933 |
| `Bodo` | 14,963 | 315.55 | 222 | 1795 |
| `Dogri` | 14,963 | 308.39 | 225 | 1974 |
| `Gujarati` | 14,963 | 313.22 | 227 | 1526 |
| `Hindi` | 14,963 | 313.66 | 230 | 2018 |
| `Kannada` | 14,963 | 316.92 | 230 | 8305 |
| `Kashmiri` | 14,963 | 339.03 | 243 | 2102 |
| `Konkani` | 14,963 | 307.78 | 227 | 1819 |
| `Maithili` | 14,963 | 305.52 | 225 | 2142 |
| `Malayalam` | 14,963 | 330.73 | 236 | 1862 |
| `Manipuri` | 14,963 | 332.17 | 234 | 2247 |
| `Marathi` | 14,963 | 312.05 | 229 | 8370 |
| `Nepali` | 14,963 | 312.62 | 229 | 1825 |
| `Oriya` | 14,963 | 312.36 | 226 | 4092 |
| `Punjabi` | 14,963 | 309.98 | 220 | 926 |
| `Sanskrit` | 14,963 | 312.86 | 225 | 1232 |
| `Santhali` | 14,963 | 334.48 | 234 | 2283 |
| `Sindhi` | 14,963 | 309.44 | 226 | 852 |
| `Tamil` | 14,963 | 331.24 | 236 | 1031 |
| `Telugu` | 14,963 | 319.9 | 232 | 911 |
| `Urdu` | 14,963 | 314.71 | 227 | 921 |

## Sample Example

**Subset**: `Assamese`

```json
{
  "input": [
    {
      "id": "4ac474ca",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nইমিউনজনিত বিকাৰসমূহৰ ভিতৰত আছে .....\n\nA) অতিরিক্ত সংবেদনশীলতা\nB) স্বয়ং-প্রতিরোধ ক্ষমতা জনিত ৰোগ\nC) রোগ প্রতিরোধ ক্ষমতাৰ অভাৱ\nD) এই সকলোবোৰ।"
    }
  ],
  "choices": [
    "অতিরিক্ত সংবেদনশীলতা",
    "স্বয়ং-প্রতিরোধ ক্ষমতা জনিত ৰোগ",
    "রোগ প্রতিরোধ ক্ষমতাৰ অভাৱ",
    "এই সকলোবোৰ।"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "Assamese",
    "topic": "Kayachikitsa"
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
    --datasets bhasha_bench_multi_ayur \
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
    datasets=['bhasha_bench_multi_ayur'],
    dataset_args={
        'bhasha_bench_multi_ayur': {
            # subset_list: ['Assamese', 'Bengali', 'Bodo']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
