# BhashaBench-Multi (Finance)


## Overview

BhashaBench-Multi (Finance) is a domain-specific multiple-choice benchmark evaluating LLM knowledge
of finance across 22 Indic languages. Each question originates in English and is machine
translated (with LLM-judged translation quality scores) into the target language; this adapter uses
the translated question/choices.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: A finance question with 4 answer choices, in one of 22 Indic languages
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
| **Benchmark Name** | `bhasha_bench_multi_finance` |
| **Dataset ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

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
    --datasets bhasha_bench_multi_finance \
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
    datasets=['bhasha_bench_multi_finance'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
