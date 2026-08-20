# BhashaBench-V1 (Krishi)


## Overview

BhashaBench-Krishi is the predecessor of BhashaBench-Multi's krishi domain: a domain-specific
multiple-choice benchmark evaluating LLM knowledge of agriculture (Krishi), covering English and Hindi.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: An agriculture (Krishi) question with 4 answer choices, in English or Hindi
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
- For broader language coverage of the same domain, see `bhasha_bench_multi_krishi`
  (22 Indic languages, not gated)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bhashabenchv1_krishi` |
| **Dataset ID** | [bharatgenai/BhashaBench-Krishi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Krishi/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 15,405 |
| Prompt Length (Mean) | 409.45 chars |
| Prompt Length (Min/Max) | 223 / 1841 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `English` | 12,648 | 429.18 | 223 | 1841 |
| `Hindi` | 2,757 | 318.93 | 233 | 678 |

## Sample Example

**Subset**: `English`

```json
{
  "input": [
    {
      "id": "afa2a6e0",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nIt is state or condition of atmosphere at given place and given time.?\n\nA) Climate\nB) Weather\nC) Environment\nD) Atmosphere"
    }
  ],
  "choices": [
    "Climate",
    "Weather",
    "Environment",
    "Atmosphere"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English",
    "topic": ""
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
    --datasets bhashabenchv1_krishi \
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
    datasets=['bhashabenchv1_krishi'],
    dataset_args={
        'bhashabenchv1_krishi': {
            # subset_list: ['English', 'Hindi']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
