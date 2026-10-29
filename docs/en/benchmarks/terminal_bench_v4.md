# Terminal-Bench-4.0


## Overview

Terminal-Bench 4.0 evaluates agents on 66 challenging terminal tasks using the official Harbor harness. It calibrates task resources, fixes task instructions and verifiers, and removes tasks that no longer provide useful differentiation.

## Task Description

- **Task Type**: Command-Line Agent Evaluation
- **Input**: Task instructions and an isolated runtime environment
- **Output**: Agent-produced artifacts and task completion, checked by the official verifier
- **Domain**: Software engineering, machine learning, security, science, and operations

## Key Features

- Fixed upstream dataset revision **4.0.0**, with content-addressed task versions
- 66 tasks, including 11 multi-container tasks and 3 tasks requiring an H100 GPU
- Separate verifier environments for all tasks, including a task with a GPU verifier
- Official agent timeout of 8 hours per task; task-specific CPU, memory, and verifier budgets
- Configurable Harbor agents, explicit task-name filtering, and repeated evaluation

## Evaluation Notes

- Requires **Python>=3.12**, **Harbor>=0.14.0,<1.0.0**, and `pip install 'evalscope[terminal_bench]'`
- Scores are the mean of valid official verifier rewards (0/1); failed trials and invalid rewards are not valid scores
- Defaults to Docker and terminus-2; terminus-2 uses the configured EvalScope model, while external CLI agents use their own API credentials
- Full evaluation requires a Docker host with H100 GPU access and multi-container support; Modal is an alternative with `harbor[modal]` and account credentials
- Task resources and timeouts are inherited from upstream; max_turns defaults to None, using the Harbor agent default
- Use task_names for an explicit debugging subset and repeats for repeated trials; subset results and results from older benchmark versions are not directly comparable to a full 4.0 evaluation
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/terminal_bench.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `terminal_bench_v4` |
| **Dataset ID** | [4](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 66 |
| Prompt Length (Mean) | 0 chars |
| Prompt Length (Min/Max) | 0 / 0 chars |

## Sample Example

**Subset**: `test`

```json
{
  "input": [
    {
      "id": "5706a469",
      "content": ""
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "path": null,
    "git_url": null,
    "git_commit_id": null,
    "name": "terminal-bench/layout-config-recreation2",
    "ref": "sha256:193312b14b1dcb6ef551efff691cbb708b2032f58879b0b19efe88a2cddc9deb",
    "overwrite": false,
    "download_dir": "~/.cache/evalscope/terminal_bench_v4",
    "source": "terminal-bench/terminal-bench"
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
| `environment_type` | `str` | `docker` | Environment type for running the benchmark. Choices: ['docker', 'daytona', 'e2b', 'modal'] |
| `agent_name` | `str` | `terminus-2` | Agent type to be used in Harbor. Only terminus-2 uses the evalscope model for inference; other agents (claude-code, codex, etc.) run as standalone CLI tools with their own API keys. Choices: ['oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'] |
| `timeout_multiplier` | `float` | `1.0` | Timeout multiplier. If timeout errors occur, consider increasing this value. |
| `agent_timeout_sec` | `float` | `None` | Final agent timeout in seconds. Cannot be combined with agent_timeout_multiplier. |
| `verifier_timeout_sec` | `float` | `None` | Final verifier timeout in seconds. Cannot be combined with verifier_timeout_multiplier. |
| `agent_timeout_multiplier` | `float` | `None` | Agent timeout multiplier. Overrides timeout_multiplier for the agent phase. |
| `verifier_timeout_multiplier` | `float` | `None` | Verifier timeout multiplier. Overrides timeout_multiplier for the verifier phase. |
| `max_turns` | `int` | `None` | Maximum agent turns. None uses the Harbor agent default. |
| `environment_kwargs` | `dict` | `{}` | Extra kwargs passed to Harbor EnvironmentConfig. Supported keys: override_cpus, override_memory_mb, override_storage_mb, override_gpus, force_build, delete, env, etc. |
| `task_names` | `list` | `None` | Optional Harbor task names or glob patterns, e.g. terminal-bench/ctr-optimization. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v4 \
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
    datasets=['terminal_bench_v4'],
    dataset_args={
        'terminal_bench_v4': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
