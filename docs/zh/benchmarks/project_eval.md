# Project Eval

## Overview
Evaluate externally generated business-agent eval sets with rubric judging and read-only end-state checks.

## Task Description
- **Task Type**: Business-agent conversation with optional multi-turn user scripts.
- **Input**: A JSONL task with a question or messages, reference answer, rubric and read-only checks.
- **Output**: Agent reply, trace, rubric verdicts and end-state check evidence.
- **Domain**: Enterprise business agents with local or remote business state.

## Key Features
- Consumes externally authored eval sets; EvalScope does not read project source.
- Supports SQLite, HTTP/JSON, JSON file and file checks.
- Supports L0 and L2 inline trace; process criteria are not evaluable without trace.

## Evaluation Notes
- Metrics: task_pass, rubric_score and evidence_coverage; task_pass is primary.
- Checks are read-only; a failed must-pass outcome check short-circuits the task.
- The feature is gated behind the project_eval service flags.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `project_eval` |
| **Dataset ID** | `project_eval` |
| **Paper** | N/A |
| **Tags** | N/A |
| **Metrics** | `task_pass`, `rubric_score`, `evidence_coverage` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `N/A` |


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
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets project_eval \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
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
    datasets=['project_eval'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
