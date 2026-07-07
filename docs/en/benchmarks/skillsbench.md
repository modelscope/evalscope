# SkillsBench


## Overview

SkillsBench evaluates whether coding agents can discover and apply task-bundled Agent Skills. Each task contains an
instruction, an optional skill directory, a Docker environment, an oracle solution, and a verifier. EvalScope builds the
task Docker image, runs the selected agent or oracle in that image, then executes the task verifier.

## Task Description

- **Task Type**: Agent skill usage / tool-assisted task completion
- **Input**: The natural-language task prompt from `task.md`
- **Output**: Files or state changes produced inside the task container, scored by the task verifier
- **Dataset**: Local SkillsBench task repository supplied through `extra_params.tasks_dir`
- **Environment**: Per-task Docker image built from `environment/Dockerfile`
- **Skills**: Optional task-bundled skills from `environment/skills`
- **Metric**: `score` from `/logs/verifier/reward.txt`; `success` is 1 when `score > 0`

## Key Features

- Builds or reuses a content-hashed Docker image for each selected task.
- Runs the task in `no-skill` or `with-skill` mode without mixing the two conditions in one EvalScope run.
- Injects task-bundled skills through EvalScope's agent skill runtime instead of baking runner-specific skill paths into
  the image.
- Supports EvalScope native agents and external agent runners through the shared agent environment interface.
- Saves verifier stdout, reward, and optional CTRF artifacts under the run output directory.

## Evaluation Notes

- Default `skill_mode` is `no-skill`.
- Run `no-skill` and `with-skill` separately, then compare runs with EvalScope's run comparison tools.
- `self-gen` and `tasks-extra` are not supported by this adapter version.
- `runner='oracle'` runs the official `oracle/solve.sh` for smoke testing; agent runs require `agent_config`.
- The verifier executes `verifier/test.sh` and may install dependencies from the network.

## Scoring and Comparison

- `score` is the verifier reward parsed from `/logs/verifier/reward.txt`.
- `success` is derived locally as `1.0` when `score > 0`, otherwise `0.0`.
- EvalScope does not automatically compute the no-skill versus with-skill delta; run both modes and compare their reports.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `skillsbench` |
| **Dataset ID** | `skillsbench` |
| **Paper** | N/A |
| **Tags** | `Agent`, `MultiTurn` |
| **Metrics** | `score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `N/A` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks_dir` | `str` | `` | Path to the SkillsBench tasks directory. |
| `task_ids` | `list` | `[]` | Optional list of task ids to run. |
| `skill_mode` | `str` | `no-skill` | SkillsBench skill mode. Choices: ['no-skill', 'with-skill'] |
| `force_rebuild` | `bool` | `False` | Force rebuilding task Docker images. |
| `agent_timeout_sec` | `float` | `None` | Override task agent timeout. |
| `verifier_timeout_sec` | `float` | `None` | Override task verifier timeout. |
| `runner` | `str` | `agent` | Use "oracle" to run oracle/solve.sh instead of an agent. Choices: ['agent', 'oracle'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets skillsbench \
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
    datasets=['skillsbench'],
    dataset_args={
        'skillsbench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


