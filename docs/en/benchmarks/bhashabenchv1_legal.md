# BhashaBench-V1 (Legal)


## Overview

BhashaBench-Legal is the predecessor of BhashaBench-Multi's legal domain: a domain-specific
multiple-choice benchmark evaluating LLM knowledge of Indian law, covering English and Hindi.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: An Indian law question with 4 answer choices, in English or Hindi
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
- For broader language coverage of the same domain, see `bhasha_bench_multi_legal`
  (22 Indic languages, not gated)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bhashabenchv1_legal` |
| **Dataset ID** | [bharatgenai/BhashaBench-Legal](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Legal/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 24,365 |
| Prompt Length (Mean) | 513.88 chars |
| Prompt Length (Min/Max) | 229 / 4628 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `English` | 17,047 | 539.36 | 233 | 4628 |
| `Hindi` | 7,318 | 454.52 | 229 | 1748 |

## Sample Example

**Subset**: `English`

```json
{
  "input": [
    {
      "id": "6e1ae42b",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nPower to amend the issue or frame additional issues prior to passing of a decree vests in a Court by virtue of which provision of the Code of Civil Procedure, 1908?\n\nA) Order XIV Rule 1\nB) Order XIV Rule 5\nC) Order XIV Rule 6\nD) Section 151"
    }
  ],
  "choices": [
    "Order XIV Rule 1",
    "Order XIV Rule 5",
    "Order XIV Rule 6",
    "Section 151"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English",
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
    --datasets bhashabenchv1_legal \
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
    datasets=['bhashabenchv1_legal'],
    dataset_args={
        'bhashabenchv1_legal': {
            # subset_list: ['English', 'Hindi']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
