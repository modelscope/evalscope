# ANLI


## Overview

Adversarial natural language inference over three independently collected rounds.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Premise and hypothesis
- **Output**: Entailment, neutral or contradiction
- **Domain**: English language understanding

## Key Features

- Public dataset: `facebook/anli` on ModelScope
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Reports accuracy per round and a sample-weighted overall score. Test explanations are excluded from model inputs.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.0


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `anli` |
| **Dataset ID** | [facebook/anli](https://modelscope.cn/datasets/facebook/anli/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,200 |
| Prompt Length (Mean) | 785.53 chars |
| Prompt Length (Min/Max) | 541 / 1303 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `r1` | 1,000 | 791.1 | 684 | 987 |
| `r2` | 1,000 | 790.79 | 681 | 1015 |
| `r3` | 1,200 | 776.5 | 541 | 1303 |

## Sample Example

**Subset**: `r1`

```json
{
  "input": [
    {
      "id": "00a00842",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C.\n\nPremise:\nErnest Jones is a British jeweller and watchma ... [TRUNCATED 261 chars] ... ones store was opened on the continent of Europe.\n\nA) Entailment: the hypothesis follows from the premise.\nB) Neutral: the premise does not determine whether the hypothesis is true.\nC) Contradiction: the hypothesis conflicts with the premise."
    }
  ],
  "choices": [
    "Entailment: the hypothesis follows from the premise.",
    "Neutral: the premise does not determine whether the hypothesis is true.",
    "Contradiction: the hypothesis conflicts with the premise."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "uid": "4aae63a8-fcf7-406c-a2f3-50c31c5934a9",
    "round": "r1"
  }
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

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets anli \
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
    datasets=['anli'],
    dataset_args={
        'anli': {
            # subset_list: ['r1', 'r2', 'r3']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
