# Terminal-Bench-2.0


## Overview

Terminal-Bench v2 is a command-line benchmark suite that evaluates AI agents on 89 real-world, multi-step terminal tasks. Tasks range from compiling and debugging to system administration, running within isolated containers with rigorous validation.

## Task Description

- **Task Type**: Command-Line Agent Evaluation
- **Input**: Terminal task specification
- **Output**: Task completion via agent actions
- **Domains**: System administration, compilation, debugging, file operations

## Key Features

- 89 real-world terminal tasks
- Multi-step task completion requirements
- Isolated container execution environment
- Binary scoring (0/1) with auto-validation
- Multiple agent types supported (terminus-2, claude-code, codex, etc.)

## Evaluation Notes

- Requires **Python>=3.12** and `pip install harbor==0.1.28`
- Environment options: docker, daytona, e2b, modal
- Configurable agent types and timeout settings
- Maximum turns configurable (default: 200)
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/terminal_bench.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `terminal_bench_v2` |
| **Dataset ID** | [terminal-bench-2.git](https://github.com/laude-institute/terminal-bench-2.git) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


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
| `environment_type` | `str` | `docker` | Environment type for running the benchmark. Choices: ['docker', 'daytona', 'e2b', 'modal'] |
| `agent_name` | `str` | `terminus-2` | Agent type to be used in Harbor. Choices: ['oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'] |
| `timeout_multiplier` | `float` | `1.0` | Timeout multiplier. If timeout errors occur, consider increasing this value. |
| `max_turns` | `int` | `200` | Maximum number of turns for the agent to complete the task. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v2 \
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
    datasets=['terminal_bench_v2'],
    dataset_args={
        'terminal_bench_v2': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


