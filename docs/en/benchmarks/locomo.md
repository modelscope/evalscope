# LoCoMo


## Overview

LoCoMo evaluates very long-term conversational memory in two-person multi-session dialogues. This adapter supports the
official question-answering task from `locomo10.json`.

## Task Description

- **Task Type**: Long-context question answering
- **Input**: Multi-session conversation history with session dates and a question
- **Output**: Short free-form answer
- **Subsets**: `qa`

## Key Features

- Uses the official LoCoMo QA data file hosted on ModelScope
- Supports full-history long-context prompts and evidence-only oracle prompts
- Includes image captions from the released data when present, but does not download image files
- Uses LoCoMo's rule-based F1 / adversarial refusal scoring instead of an LLM judge

## Evaluation Notes

- Default subset is `qa` with `eval_mode=long_context`
- Use `extra_params.eval_mode='oracle_context'` for evidence-only upper-bound evaluation
- Event summarization, multimodal dialog generation, and RAG retrieval are not included in this QA adapter


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `locomo` |
| **Dataset ID** | [evalscope/locomo](https://modelscope.cn/datasets/evalscope/locomo/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `MultiTurn`, `QA` |
| **Metrics** | `f1` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,986 |
| Prompt Length (Mean) | 94097.72 chars |
| Prompt Length (Min/Max) | 55178 / 111026 chars |

## Sample Example

**Subset**: `qa`

```json
{
  "input": [
    {
      "id": "b585f7b0",
      "content": "Below is a conversation between two people: Caroline and Melanie. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\nDATE: 1:56 pm on 8 May, 2023\nCONVERSATION:\nC ... [TRUNCATED 74397 chars] ... m of a short phrase for the following question. Answer with exact words from the context whenever possible.\n\nQuestion: When did Caroline go to the LGBTQ support group? Use DATE of CONVERSATION to answer with an approximate date. Short answer:"
    }
  ],
  "target": "7 May 2023",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "sample_id": "conv-26",
    "qa_index": 0,
    "category": 2,
    "category_name": "temporal",
    "raw_question": "When did Caroline go to the LGBTQ support group?",
    "answer": "7 May 2023",
    "adversarial_answer": null,
    "eval_mode": "long_context",
    "question": "When did Caroline go to the LGBTQ support group? Use DATE of CONVERSATION to answer with an approximate date.",
    "evidence": [
      "D1:3"
    ]
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `long_context` | Evaluation mode: long_context or oracle_context. Choices: ['long_context', 'oracle_context'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets locomo \
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
    datasets=['locomo'],
    dataset_args={
        'locomo': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


