# Terminal-Bench-2.1


## Overview

Terminal-Bench v2.1 is an improved iteration of Terminal-Bench 2.0, with 26 task fixes addressing bugs, timeout adjustments, and reward hacking prevention. Recommended over v2.0 for new evaluations.

## Task Description

- **Task Type**: Command-Line Agent Evaluation
- **Input**: Terminal task specification
- **Output**: Task completion via agent actions
- **Domains**: System administration, compilation, debugging, file operations

## Key Features

- Verified iteration of Terminal-Bench 2.0 with 26 task fixes
- Improved robustness against reward hacking
- Isolated container execution environment
- Binary scoring (0/1) with auto-validation
- Multiple agent types supported (terminus-2, claude-code, codex, etc.)

## Evaluation Notes

- Requires **Python>=3.12** and `pip install evalscope[terminal_bench]`
- Environment options: docker, daytona, e2b, modal
- Configurable agent types and timeout settings
- Maximum turns configurable (default: 200)
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/terminal_bench.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `terminal_bench_v2_1` |
| **Dataset ID** | [latest](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 89 |
| Prompt Length (Mean) | 0 chars |
| Prompt Length (Min/Max) | 0 / 0 chars |

## Sample Example

**Subset**: `test`

```json
{
  "input": [
    {
      "id": "d3a63987",
      "content": ""
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "path": null,
    "git_url": null,
    "git_commit_id": null,
    "name": "terminal-bench/write-compressor",
    "ref": "sha256:d9ddd9a8e925e2c566b37b2492cbf995afecefe58874e4043ef78d7f3c892c7e",
    "overwrite": false,
    "download_dir": "/Users/yunlin/.cache/evalscope/terminal_bench_v2_1",
    "source": "terminal-bench/terminal-bench-2-1"
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
| `max_turns` | `int` | `200` | Maximum number of turns for the agent to complete the task. |
| `environment_kwargs` | `dict` | `{}` | Extra kwargs passed to Harbor EnvironmentConfig. Supported keys: override_cpus, override_memory_mb, override_storage_mb, override_gpus, force_build, delete, env, etc. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v2_1 \
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
    datasets=['terminal_bench_v2_1'],
    dataset_args={
        'terminal_bench_v2_1': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


