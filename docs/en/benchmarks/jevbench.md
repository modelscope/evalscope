# JevBench public subset


## Overview

Public 231 Benchmark Heaven decisions, preserving original rubrics and labels.

## Task Description

- **Task Type**: Typed text decisions
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- Pinned Benchmark Heaven public snapshot: 72 original, 48 easy and 111 hard tasks (139 Choice, 74 Noul, 18 Score)
- Source: [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench/tree/609a664e81f46e03731600bd7bd9f1b7a61257c5); MIT-licensed public records
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Zero-shot deterministic grading; gold answers and provenance never enter requests
- Primary: public-subset accuracy (missing predictions count as wrong); also family-macro accuracy, per-kind probability metrics and Score expected-value errors
- Four- and five-level Score tasks keep their original rubrics; AutoTrust fixed-six-level services reject these tasks explicitly
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- Public scorer: probability sum tolerance 0.001; normalize only above that within 0.02; Noul ties select no
- This public snapshot does not produce the full JevBench Composite or Capability leaderboard metrics


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `jevbench` |
| **Dataset ID** | [evalscope/jevbench](https://modelscope.cn/datasets/evalscope/jevbench/summary) |
| **Paper** | N/A |
| **Tags** | `Decision` |
| **Metrics** | `public_accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 231 |
| Prompt Length (Mean) | 2521.31 chars |
| Prompt Length (Min/Max) | 224 / 16126 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "f98fca19",
      "content": "{\"state\": \"Where is my package? I ordered it last week and it still hasn't arrived.\", \"question\": {\"criteria\": {\"billing_question\": \"Asks about a charge, invoice or payment\", \"cancel_order\": \"Wants to cancel an order\", \"change_address\": \"Wants to change the delivery address\", \"report_damage\": \"Received an item that is broken or damaged\", \"track_order\": \"Wants to know where an order is or when it arrives\"}, \"instructions\": \"Which intent does the user's message express?\", \"type\": \"choice\"}}",
      "internal": {
        "systemone": {
          "kind": "choice",
          "state": "Where is my package? I ordered it last week and it still hasn't arrived.",
          "instructions": "Which intent does the user's message express?",
          "criteria": {
            "billing_question": "Asks about a charge, invoice or payment",
            "cancel_order": "Wants to cancel an order",
            "change_address": "Wants to change the delivery address",
            "report_damage": "Received an item that is broken or damaged",
            "track_order": "Wants to know where an order is or when it arrives"
          },
          "answer_prefix": "ANSWER: "
        }
      }
    }
  ],
  "target": "track_order",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source_id": "easy-intent-00",
    "decision_gold": "track_order",
    "decision_kind": "choice",
    "source_group": null,
    "family": "intent",
    "tier": "",
    "language": "",
    "continuous": false,
    "decision_labels": [
      "billing_question",
      "cancel_order",
      "change_address",
      "report_damage",
      "track_order"
    ],
    "fixed_classes": false
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --eval-type systemone_api \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets jevbench \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['jevbench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
