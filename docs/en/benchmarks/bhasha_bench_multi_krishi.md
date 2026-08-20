# BhashaBench-Multi (Krishi)


## Overview

BhashaBench-Multi (Krishi) is a domain-specific multiple-choice benchmark evaluating LLM knowledge
of agriculture (Krishi) across 22 Indic languages. Each question originates in English and is machine
translated (with LLM-judged translation quality scores) into the target language; this adapter uses
the translated question/choices.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: A agriculture (Krishi) question with 4 answer choices, in one of 22 Indic languages
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
| **Benchmark Name** | `bhasha_bench_multi_krishi` |
| **Dataset ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 338,910 |
| Prompt Length (Mean) | 411.84 chars |
| Prompt Length (Min/Max) | 207 / 2882 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Assamese` | 15,405 | 402.14 | 224 | 2265 |
| `Bengali` | 15,405 | 406.38 | 224 | 1506 |
| `Bodo` | 15,405 | 417.81 | 207 | 2186 |
| `Dogri` | 15,405 | 403.83 | 220 | 1988 |
| `Gujarati` | 15,405 | 397.6 | 224 | 1380 |
| `Hindi` | 15,405 | 407.8 | 224 | 1572 |
| `Kannada` | 15,405 | 407.21 | 224 | 1407 |
| `Kashmiri` | 15,405 | 442.73 | 245 | 2668 |
| `Konkani` | 15,405 | 402.57 | 222 | 1969 |
| `Maithili` | 15,405 | 393.7 | 224 | 1783 |
| `Malayalam` | 15,405 | 429.89 | 224 | 1661 |
| `Manipuri` | 15,405 | 436.33 | 240 | 2882 |
| `Marathi` | 15,405 | 406.47 | 224 | 1520 |
| `Nepali` | 15,405 | 402.05 | 224 | 1440 |
| `Oriya` | 15,405 | 392.2 | 221 | 1366 |
| `Punjabi` | 15,405 | 404.2 | 222 | 1536 |
| `Sanskrit` | 15,405 | 411.22 | 224 | 1412 |
| `Santhali` | 15,405 | 440.59 | 234 | 2773 |
| `Sindhi` | 15,405 | 392.36 | 224 | 1233 |
| `Tamil` | 15,405 | 441.53 | 224 | 2165 |
| `Telugu` | 15,405 | 412.75 | 224 | 1432 |
| `Urdu` | 15,405 | 409.12 | 224 | 2132 |

## Sample Example

**Subset**: `Assamese`

```json
{
  "input": [
    {
      "id": "70df7242",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nইয়াকোনো বিশেষ স্থান আৰু সময়ত বায়ুমণ্ডলৰ অৱস্থা বা পৰিস্থিতি বুলি কোৱা হয়।\n\nA) জলবায়ু\nB) আবহাওয়া\nC) পৰ্যাৱৰণ\nD) বায়ুমণ্ডল"
    }
  ],
  "choices": [
    "জলবায়ু",
    "আবহাওয়া",
    "পৰ্যাৱৰণ",
    "বায়ুমণ্ডল"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "Assamese",
    "topic": null
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
    --datasets bhasha_bench_multi_krishi \
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
    datasets=['bhasha_bench_multi_krishi'],
    dataset_args={
        'bhasha_bench_multi_krishi': {
            # subset_list: ['Assamese', 'Bengali', 'Bodo']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
