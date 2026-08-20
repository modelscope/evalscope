# BhashaBench-V1 (Ayurveda)


## Overview

BhashaBench-Ayur is the predecessor of BhashaBench-Multi's ayur domain: a domain-specific
multiple-choice benchmark evaluating LLM knowledge of Ayurvedic medicine, covering English and Hindi.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: An Ayurvedic medicine question with 4 answer choices, in English or Hindi
- **Output**: Correct answer letter
- **Languages**: English, Hindi

## Key Features

- 5,600–17,000 questions per language, covering English and Hindi only
- Predecessor of BhashaBench-Multi: same domains, narrower language coverage
- Each domain is a separate repository, with English and Hindi as separate configs

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate a single language (e.g., `['Hindi']`)
- Requires access to this gated dataset - on ModelScope (the default hub), accept the terms and
  ensure you're logged in; alternatively, set `dataset_hub` to `huggingface` and use `HF_TOKEN`
  after accepting the terms on huggingface.co
- For broader language coverage of the same domain, see `bhasha_bench_multi_ayur`
  (22 Indic languages, not gated)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bhashabenchv1_ayur` |
| **Dataset ID** | [bharatgenai/BhashaBench-Ayur](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Ayur/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 14,963 |
| Prompt Length (Mean) | 307.52 chars |
| Prompt Length (Min/Max) | 222 / 1060 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `English` | 9,348 | 310.35 | 227 | 1060 |
| `Hindi` | 5,615 | 302.81 | 222 | 938 |

## Sample Example

**Subset**: `English`

```json
{
  "input": [
    {
      "id": "6a6db8b7",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nImmune disorders include .....,\n\nA) Hypersensitivity\nB) auto-immune diseases\nC) immunodeficiency\nD) all of these"
    }
  ],
  "choices": [
    "Hypersensitivity",
    "auto-immune diseases",
    "immunodeficiency",
    "all of these"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English",
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
    --datasets bhashabenchv1_ayur \
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
    datasets=['bhashabenchv1_ayur'],
    dataset_args={
        'bhashabenchv1_ayur': {
            # subset_list: ['English', 'Hindi']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
