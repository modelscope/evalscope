# General Decision


## Overview

Evaluate local JSONL datasets containing one Choice, Noul or Score question per record.

## Task Description

- **Task Type**: Text decisions: Choice, Noul and Score
- **Input**: State and one question with named options or ordered levels
- **Output**: Choice, probability of yes, or ordinal score distribution
- **Domain**: Text understanding and decision making

## Key Features

- Reads `default_test.jsonl` from a directory passed through `dataset_args.general_decision.local_path`
- Fields: `id`, `state`, `question` (type/instructions/criteria), `expected`; one question per record
- Preserves option order, source IDs and rubric descriptions
- Supports System One and native AutoTrust APIs within their declared capabilities

## Evaluation Notes

- Rule-based zero-shot grading; reference answers and provenance are used only for scoring, never as request fields
- Default primary metric: accuracy on discrete reference labels; continuous Score runs can select MAE explicitly
- Probability metrics are separated by type; Score retains its expectation and normalized RPS
- Invalid responses are excluded from ordinary metrics and reflected in execution coverage
- This is a local data entry point; no upload is required
- See [Custom decision datasets](../advanced_guides/custom_dataset/decision.md) for runnable JSONL examples,
  required and optional fields, question types, metrics, continuous scores, and configuration
- See the [decision evaluation best practice](../best_practice/decision_evaluation.md) for deployment and Web UI


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `general_decision` |
| **Dataset ID** | `general_decision` |
| **Paper** | N/A |
| **Tags** | `Custom`, `Decision` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

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
    --datasets general_decision \
    --dataset-args '{"general_decision": {"local_path": "/path/to/data"}}' \
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
    datasets=['general_decision'],
    dataset_args={
        'general_decision': {
            'local_path': '/path/to/data',  # contains default_test.jsonl
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
