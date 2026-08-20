# Sanskriti


## Overview

Sanskriti is a multiple-choice trivia benchmark testing knowledge of Indian states' culture, history,
and geography, sourced from state-specific attributes (art, cuisine, festivals, etc.) with
Wikipedia-backed answers. From the SANSKRITI paper (arXiv:2506.15355); this adapter loads the
dataset mirrored to ModelScope as `evalscope/Sanskriti`.

## Task Description

- **Task Type**: Multiple-Choice Trivia Question Answering
- **Input**: A question about a specific Indian state's culture/geography/history, with 4 answer choices
- **Output**: Correct answer letter
- **Subsets**: `association` (state-attribute association trivia), `country` (country-level trivia),
  `gk` (general knowledge), `states` (state-identification trivia)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (the dataset's only split, named `train` upstream
  despite being evaluation data)
- Questions and choices are in English
- The paper acknowledges some questions involve ambiguous cultural elements; a small number of rows
  (~0.6%) whose `answer` doesn't match any of the 4 listed options are skipped at load time


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `sanskriti` |
| **Dataset ID** | [evalscope/Sanskriti](https://modelscope.cn/datasets/evalscope/Sanskriti/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 21,726 |
| Prompt Length (Mean) | 322.93 chars |
| Prompt Length (Min/Max) | 256 / 636 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `association` | 5,453 | 343.41 | 273 | 523 |
| `country` | 5,563 | 284.48 | 256 | 417 |
| `gk` | 5,328 | 346.94 | 263 | 547 |
| `states` | 5,382 | 318.17 | 260 | 636 |

## Sample Example

**Subset**: `association`

```json
{
  "input": [
    {
      "id": "0629b222",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich of the given regions is home to the Jarawa body painting?\n\nA) Surguja district\nB) South Andaman and Middle Andaman Islands\nC) Buddha Marg, Patna\nD) Telangana"
    }
  ],
  "choices": [
    "Surguja district",
    "South Andaman and Middle Andaman Islands",
    "Buddha Marg, Patna",
    "Telangana"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "association",
  "metadata": {
    "state": "Andaman_and_Nicobar",
    "attribute": "Art"
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
    --datasets sanskriti \
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
    datasets=['sanskriti'],
    dataset_args={
        'sanskriti': {
            # subset_list: ['association', 'country', 'gk']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
