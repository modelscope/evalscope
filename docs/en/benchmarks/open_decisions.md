# OpenDecisions


## Overview

OpenDecisions v1.0: 2352 typed decisions. The upstream train split is evaluation-only.

## Task Description

- **Task Type**: Typed text decisions
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- 1178 Choice, 586 Noul and 588 Score tasks; multilingual inputs
- Dataset: [jgeuter/OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions); data CC BY-SA 4.0 with per-item provenance, scorer MIT
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Zero-shot deterministic grading; gold answers and provenance never enter requests
- Primary: Capability; official Intelligence and Calibration use the pinned v1.0 scorer at revision `5026a13ad6e5a8606810a3a5c4e410d8d6638e71`
- Official values are stored as ratios (displayed as percentages); missing predictions count as wrong in Intelligence
- Ordinary probability metrics use valid predictions only. Noul threshold is 0.5 in the official protocol
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- Score rubrics have 3-6 levels. AutoTrust fixed-six-level services cannot evaluate the complete dataset
- Limit/shuffle runs score the selected items and must not be compared with a complete leaderboard run


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `open_decisions` |
| **Dataset ID** | [evalscope/OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions/summary) |
| **Paper** | N/A |
| **Tags** | N/A |
| **Metrics** | `capability` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,352 |
| Prompt Length (Mean) | 1178.54 chars |
| Prompt Length (Min/Max) | 208 / 5407 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "fe3dcf15",
      "content": "{\"state\": \"Here are three expense lines from my trip. Turn them into a JSON list with fields merchant, date, amount_usd.\\n- Blue Cafe, 2024-03-02, $14.50\\n- MetroCab, 2024-03-02, $23.00\\n- Hotel Lira, 2024-03-04, $189.90\", \"question\": {\"type\" ... [TRUNCATED 79 chars] ... .\", \"criteria\": {\"chitchat\": \"Casual small talk, no task.\", \"creative_writing\": \"Compose original text.\", \"extraction_json\": \"Pull given details into a structured JSON shape.\", \"data_analysis\": \"Compute statistics or trends over the data.\"}}}",
      "internal": {
        "systemone": {
          "kind": "choice",
          "state": "Here are three expense lines from my trip. Turn them into a JSON list with fields merchant, date, amount_usd.\n- Blue Cafe, 2024-03-02, $14.50\n- MetroCab, 2024-03-02, $23.00\n- Hotel Lira, 2024-03-04, $189.90",
          "instructions": "Classify the task type of this request for routing.",
          "criteria": {
            "chitchat": "Casual small talk, no task.",
            "creative_writing": "Compose original text.",
            "extraction_json": "Pull given details into a structured JSON shape.",
            "data_analysis": "Compute statistics or trends over the data."
          },
          "answer_prefix": "ANSWER: "
        }
      }
    }
  ],
  "target": "extraction_json",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source_id": "jb16-llm_router-9a24706d03",
    "decision_gold": "extraction_json",
    "decision_kind": "choice",
    "source_group": null,
    "family": "llm_router",
    "tier": "easy",
    "language": "en",
    "continuous": false,
    "decision_labels": [
      "chitchat",
      "creative_writing",
      "extraction_json",
      "data_analysis"
    ],
    "fixed_classes": false
  }
}
```

*Note: Some content was truncated for display.*

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
    --datasets open_decisions \
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
    datasets=['open_decisions'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
