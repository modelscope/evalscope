# BoolQ


## Overview

Binary reading comprehension using a passage and a naturally occurring question.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Passage and yes/no question
- **Output**: Yes or No
- **Domain**: English reading comprehension

## Key Features

- Public dataset: `google/boolq` on ModelScope
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Uses the public validation labels, zero-shot by default. Decision APIs use two-option Choice by default. Set `extra_params.decision_kind="noul"` for P(yes), with `decision_threshold=0.5` by default.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.1


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `boolq` |
| **Dataset ID** | [google/boolq](https://modelscope.cn/datasets/google/boolq/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,270 |
| Prompt Length (Mean) | 822.26 chars |
| Prompt Length (Min/Max) | 300 / 5043 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "47f2b621",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nPassage:\nAll biomass goes through at least some of these  ... [TRUNCATED 1152 chars] ... versity of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.\n\nQuestion: does ethanol take more energy make that produces\n\nA) Yes\nB) No"
    }
  ],
  "choices": [
    "Yes",
    "No"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decision_kind` | `str` | `choice` | Decision API question type: choice (default) or noul. Requires a decision API for noul. |
| `decision_threshold` | `float` | `0.5` | Classify Noul as yes when P(yes) is at least this threshold, in [0, 1]. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets boolq \
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
    datasets=['boolq'],
    dataset_args={
        'boolq': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
