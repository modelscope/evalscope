# LongMemEval


## Overview

LongMemEval evaluates long-term interactive memory in chat assistants. Each question is answered from a timestamped multi-session user-assistant history.

## Task Description

- **Task Type**: Long-context / retrieval-log question answering
- **Input**: Multiple dated chat sessions + current question date + question
- **Output**: Free-form answer grounded in the history
- **Subsets**: `s` (~115K-token histories), `m` (~500 sessions, large), and `oracle` (evidence sessions only)

## Key Features

- Covers single-session, multi-session, temporal reasoning, knowledge update, preference, and abstention questions
- Supports full-history long-context prompts and official retrieval-log prompts
- Uses LongMemEval's LLM judge prompts for semantic answer correctness
- Downloads only the selected JSON file from ModelScope

## Evaluation Notes

- Default subset is `s` with `eval_mode=long_context` for standard long-context evaluation
- Use `subset_list=['oracle']` and `extra_params.eval_mode='oracle_context'` for evidence-only upper-bound evaluation
- The `m` subset is large and must be requested explicitly
- `retrieval_log` mode consumes official LongMemEval retrieval logs; it does not run embedding retrieval itself


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `longmemeval` |
| **Dataset ID** | [evalscope/longmemeval-cleaned](https://modelscope.cn/datasets/evalscope/longmemeval-cleaned/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `MultiTurn`, `QA`, `Retrieval` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 500 |
| Prompt Length (Mean) | 515877.18 chars |
| Prompt Length (Min/Max) | 480743 / 540467 chars |

## Sample Example

**Subset**: `s`

```json
{
  "input": [
    {
      "id": "99d5aff3",
      "content": "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to ... [TRUNCATED 513848 chars] ... you'll be able to optimize your pantry storage space, reduce clutter, and make meal prep and cooking more efficient. Happy organizing!\"}]\n\n\nCurrent Date: 2023/05/30 (Tue) 23:40\nQuestion: What degree did I graduate with?\nAnswer (step by step):"
    }
  ],
  "target": "Business Administration",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "e47becba",
    "question_type": "single-session-user",
    "question": "What degree did I graduate with?",
    "answer": "Business Administration",
    "question_date": "2023/05/30 (Tue) 23:40",
    "answer_session_ids": [
      "answer_280352e9"
    ],
    "is_abstention": false,
    "eval_mode": "long_context",
    "retrieved_ids": [
      "sharegpt_yywfIrx_0",
      "85a1be56_1",
      "sharegpt_Jcy1CVN_0",
      "sharegpt_Cr2tc1f_0",
      "sharegpt_DGTCD7D_0",
      "f6859b48_2",
      "52c34859_1",
      "ultrachat_231069",
      "sharegpt_qRdLQvN_7",
      "ultrachat_359984",
      "... [TRUNCATED 43 more items] ..."
    ]
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `long_context` | Evaluation mode: oracle_context, long_context, or retrieval_log. Choices: ['oracle_context', 'long_context', 'retrieval_log'] |
| `retrieval_log_path` | `str | null` | `None` | Official LongMemEval retrieval log path for eval_mode=retrieval_log. |
| `retriever_type` | `str` | `flat-session` | Retrieval prompt shape for retrieval_log mode. Choices: ['flat-session', 'flat-turn'] |
| `history_format` | `str` | `json` | History rendering format. Choices: ['json', 'nl'] |
| `user_only` | `bool` | `False` | Whether to keep only user turns in history. |
| `reading_method` | `str` | `con` | Prompt reading method. `con` asks the model to extract and reason before answering. Choices: ['direct', 'con'] |
| `topk_context` | `int` | `1000` | Maximum number of history sessions or retrieved chunks included in the prompt. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets longmemeval \
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
    datasets=['longmemeval'],
    dataset_args={
        'longmemeval': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
