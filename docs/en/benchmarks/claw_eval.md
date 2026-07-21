# Claw-Eval


## Overview

Claw-Eval evaluates assistant agents on realistic personal-assistant workflows that require tool use, file and fixture
access, multimodal inputs, and simulated user interactions. EvalScope runs the pinned official Claw-Eval Python runner,
Docker sandbox, and graders while exposing each Claw-Eval task as a normal EvalScope sample for caching, repeats,
parallel execution, reporting, and dashboard trace review.

## Task Description

- **Task Type**: Agentic personal-assistant tasks with tool use, sandbox files, multimodal fixtures, and optional
  simulated user turns.
- **Dataset**: `claw-eval/Claw-Eval` on ModelScope.
- **Subsets**: `general`, `multimodal`, and `multi_turn`; the current ModelScope manifest contains 300 tasks
  (161 general, 101 multimodal, and 38 multi_turn). Use `subset_list` to select subsets.
- **Output**: Official Claw-Eval scores and JSONL traces, EvalScope sample-level reviews, grouped summary metrics, and
  dashboard-rendered agent traces.

## Evaluation Notes

- Requires Python 3.11+ and the official package installed from the pinned source commit:
  `pip install "claw-eval[sandbox,mock,web] @
  git+https://github.com/claw-eval/claw-eval.git@d3f02d4938ab0832377d90535013def2b1a2fdc0"`.
- The installed package provides the Claw-Eval runner APIs. EvalScope also caches the same pinned source archive because
  `tasks/` and `Dockerfile.agent` are runtime assets, then loads the task manifest and fixtures from ModelScope.
- Full fixtures are downloaded from ModelScope (`data/fixtures.tar.gz`) and linked into the official task tree before
  execution. The archive is large; use `limit` or `extra_params.task_ids` for smoke runs.
- Each selected Claw-Eval task is one EvalScope sample. Official scoring runs once per sample; use EvalScope `repeats`
  for repeated trials per task and `eval_batch_size` for task-level worker concurrency.
- Claw-Eval runs with the official Docker sandbox image. If `claw-eval-agent:latest` is missing locally, EvalScope
  builds it automatically from the cached official `Dockerfile.agent`. The first run can be slow.
- EvalScope `use_cache` resumes completed task-level samples. Claw-Eval trace JSONL files are stored under
  `outputs/.../claw_eval/<split>/traces` and converted to EvalScope agent traces for dashboard visualization.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `claw_eval` |
| **Dataset ID** | [claw-eval/Claw-Eval](https://modelscope.cn/datasets/claw-eval/Claw-Eval/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `MultiModal`, `MultiTurn` |
| **Metrics** | `avg_score`, `pass_at_k`, `pass_hat_k`, `error_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 300 |
| Prompt Length (Mean) | 47.36 chars |
| Prompt Length (Min/Max) | 30 / 60 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `general` | 161 | 46.47 | 34 | 58 |
| `multimodal` | 101 | 49.75 | 30 | 60 |
| `multi_turn` | 38 | 44.79 | 35 | 51 |

## Sample Example

**Subset**: `general`

```json
{
  "input": [
    {
      "id": "a34b7f6b",
      "content": "Run Claw-Eval task T001zh_email_triage."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "general",
  "metadata": {
    "task_id": "T001zh_email_triage",
    "split": "general",
    "task_name": "",
    "difficulty": "",
    "dataset_id": "claw-eval/Claw-Eval",
    "dataset_hub": "modelscope"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_ids` | `list` | `[]` | Optional exact Claw-Eval task ids to run after split filtering. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets claw_eval \
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
    datasets=['claw_eval'],
    dataset_args={
        'claw_eval': {
            # subset_list: ['general', 'multimodal', 'multi_turn']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
