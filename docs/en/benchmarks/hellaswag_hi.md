# HellaSwag-Hindi


## Overview

HellaSwag-Hindi is a Hindi translation of the HellaSwag commonsense sentence-completion benchmark's
full validation set. The context stem stays in English; the 4 candidate continuations are translated
into Hindi, so the model must connect an English scenario to its most plausible Hindi-phrased ending.
Sourced from `ai4bharat/hellaswag-translated` (the canonical name; the older `ai4bharat/hellaswag-hi`
ID redirects here), the same dataset used by lighteval's `community_hellaswag_hin` tasks.

## Task Description

- **Task Type**: Commonsense Sentence Completion (mixed-language)
- **Input**: An English context sentence with 4 Hindi-language candidate continuations
- **Output**: Correct answer letter
- **Coverage**: Full HellaSwag validation set (10,042 examples)

## Key Features

- Full HellaSwag validation set: 10,042 examples with gold labels
- English context stem paired with Hindi-translated candidate endings (mixed-language setup)
- Same dataset used by lighteval's `community_hellaswag_hin` task suite

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (validation split, the only labeled split
  available — HellaSwag's `test` split ships without gold labels)
- Loads from ModelScope by default (mirrored as `ai4bharat/hellaswag-translated`), no token required


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hellaswag_hi` |
| **Dataset ID** | [ai4bharat/hellaswag-translated](https://modelscope.cn/datasets/ai4bharat/hellaswag-translated/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 10,042 |
| Prompt Length (Mean) | 1021.77 chars |
| Prompt Length (Min/Max) | 367 / 1977 chars |

## Sample Example

**Subset**: `hi`

```json
{
  "input": [
    {
      "id": "f96132ec",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nA man is sitting on a roof. he\n\nA) वह स्की की एक जोड़ी को लपेटने के लिए रैप का उपयोग कर रहा है।\nB) यह स्तर की टाइलों को चीर रहा है।\nC) वह एक रूबिक क्यूब पकड़े हुए है।\nD) एक छत पर छत खींचना शुरू करता है।"
    }
  ],
  "choices": [
    "वह स्की की एक जोड़ी को लपेटने के लिए रैप का उपयोग कर रहा है।",
    "यह स्तर की टाइलों को चीर रहा है।",
    "वह एक रूबिक क्यूब पकड़े हुए है।",
    "एक छत पर छत खींचना शुरू करता है।"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "activity_label": "Roof shingle removal"
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
    --datasets hellaswag_hi \
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
    datasets=['hellaswag_hi'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
