# STS-B Decision


## Overview

Sentence similarity rated on six ordered levels from 0 to 5; continuous gold is preserved.

## Task Description

- **Task Type**: Typed text decisions
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- Uses the public test split of `sentence-transformers/stsb` on ModelScope
- Converts source similarity labels from [0, 1] to [0, 5], retaining fractional gold values
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Zero-shot deterministic grading; gold answers and provenance never enter requests
- Primary: MAE of the expected Score value; also cross-sample Spearman and Pearson correlations
- Does not discretize continuous gold for accuracy or ranked probability scores
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- Six levels (0-5) work with both protocols; constant or insufficient inputs leave correlations undefined


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `stsb` |
| **Dataset ID** | [sentence-transformers/stsb](https://modelscope.cn/datasets/sentence-transformers/stsb/summary) |
| **Paper** | N/A |
| **Tags** | N/A |
| **Metrics** | `mae` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,379 |
| Prompt Length (Mean) | 391.35 chars |
| Prompt Length (Min/Max) | 314 / 674 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "b8369a8a",
      "content": "{\"state\": {\"sentence1\": \"A girl is styling her hair.\", \"sentence2\": \"A girl is brushing her hair.\"}, \"question\": {\"type\": \"score\", \"instructions\": \"Rate the semantic similarity of sentence1 and sentence2.\", \"criteria\": [\"Unrelated\", \"Very different\", \"Some shared meaning\", \"Mostly similar\", \"Nearly equivalent\", \"Completely equivalent\"]}}",
      "internal": {
        "systemone": {
          "kind": "score",
          "state": {
            "sentence1": "A girl is styling her hair.",
            "sentence2": "A girl is brushing her hair."
          },
          "instructions": "Rate the semantic similarity of sentence1 and sentence2.",
          "criteria": [
            "Unrelated",
            "Very different",
            "Some shared meaning",
            "Mostly similar",
            "Nearly equivalent",
            "Completely equivalent"
          ],
          "answer_prefix": "ANSWER: "
        }
      }
    }
  ],
  "target": "2.5",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source_id": null,
    "decision_gold": 2.5,
    "decision_kind": "score",
    "source_group": null,
    "family": "",
    "tier": "",
    "language": "",
    "continuous": true,
    "decision_labels": [
      "0",
      "1",
      "2",
      "3",
      "4",
      "5"
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
    --datasets stsb \
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
    datasets=['stsb'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
