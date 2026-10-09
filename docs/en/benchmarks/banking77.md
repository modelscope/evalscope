# BANKING77


## Overview

Fine-grained banking support intent classification with the complete 77-class taxonomy.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Banking support message
- **Output**: One of 77 banking intents
- **Domain**: Customer support

## Key Features

- Public dataset: `mteb/banking77` on Hugging Face
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Reports accuracy, not macro-F1. All 77 intents remain available for every sample; no candidate pruning or added out-of-scope intent.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.0


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `banking77` |
| **Dataset ID** | [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) |
| **Paper** | N/A |
| **Tags** | `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,076 |
| Prompt Length (Mean) | 2354.28 chars |
| Prompt Length (Min/Max) | 2313 / 2668 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "627e7aaf",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,1,2,3,4,5,6 ... [TRUNCATED 1840 chars] ... e to verify identity\n44) verify my identity\n45) verify source of funds\n46) verify top up\n47) virtual card not working\n48) visa or mastercard\n49) why verify identity\n50) wrong amount of cash received\n51) wrong exchange rate for cash withdrawal"
    }
  ],
  "choices": [
    "activate my card",
    "age limit",
    "apple pay or google pay",
    "atm support",
    "automatic top up",
    "balance not updated after bank transfer",
    "balance not updated after cheque or cash deposit",
    "beneficiary not allowed",
    "cancel transfer",
    "card about to expire",
    "... [TRUNCATED 67 more items] ..."
  ],
  "target": "L",
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

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets banking77 \
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
    datasets=['banking77'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
