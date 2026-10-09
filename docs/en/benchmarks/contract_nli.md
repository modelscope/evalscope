# ContractNLI (classification)


## Overview

Document-level natural language inference on complete non-disclosure agreements.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Full contract and hypothesis
- **Output**: Entailment, contradiction or not mentioned
- **Domain**: Legal document understanding

## Key Features

- Public dataset: `tasksource/contract-nli` on Hugging Face
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Uses the full-document contractnli_b data-only mirror. Each contract-hypothesis pair is one sample. Reports classification accuracy only; evidence extraction and official F1 metrics are not evaluated. Contract and hypothesis hashes identify related samples.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.0


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `contract_nli` |
| **Dataset ID** | [tasksource/contract-nli](https://huggingface.co/datasets/tasksource/contract-nli) |
| **Paper** | N/A |
| **Tags** | `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,091 |
| Prompt Length (Mean) | 11719.39 chars |
| Prompt Length (Min/Max) | 1701 / 42345 chars |

## Sample Example

**Subset**: `contractnli_b`

```json
{
  "input": [
    {
      "id": "499ba6f1",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C.\n\nContract:\nNON-DISCLOSURE AGREEMENT\nRequired under JEA's ... [TRUNCATED 16664 chars] ... body Disclosing Party's Confidential Information.\n\nA) Contradiction: the contract contradicts the hypothesis.\nB) Entailment: the contract supports the hypothesis.\nC) Not mentioned: the contract neither supports nor contradicts the hypothesis."
    }
  ],
  "choices": [
    "Contradiction: the contract contradicts the hypothesis.",
    "Entailment: the contract supports the hypothesis.",
    "Not mentioned: the contract neither supports nor contradicts the hypothesis."
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "contract_id": "944f3cc63d215fdeb4182cbeb0d4b6ed3f3a5dcc51fe0680c879e709cabc0502",
    "hypothesis_id": "e1ce8a23b0a81e1223e43ac59ff5c4eae4389b816b2b08c8803b5c02c1f94eca"
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
    --datasets contract_nli \
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
    datasets=['contract_nli'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
