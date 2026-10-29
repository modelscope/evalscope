# Terminal-Bench

## Introduction

Terminal-Bench is a command-line benchmark suite designed to evaluate AI agents on real-world multi-step terminal tasks. These tasks cover a wide range of scenarios from compilation and debugging to system administration, all executed in isolated containers. Each task is rigorously validated and automatically scored (0/1), aiming to push frontier models to prove they can not only answer questions but also take action.

EvalScope supports three versions:

- **`terminal_bench_v2`**: The original 89-task benchmark (Terminal-Bench 2.0).
- **`terminal_bench_v2_1`**: An improved iteration with 26 task fixes addressing bugs, timeout adjustments, and reward hacking prevention (Terminal-Bench 2.1, recommended).
- **`terminal_bench_v4`**: Terminal-Bench 4.0, pinned to upstream revision **4.0.0**, with 66 tasks and separate verifier environments. Results are not directly comparable to 2.0/2.1.

Links:

- Project Repository: [harbor](https://github.com/harbor-framework/harbor)
- Dataset (Hub): [terminal-bench-2](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2/latest) | [terminal-bench-2-1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest) | [terminal-bench 4.0](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4)

## Requirements

```{important}
- **Python Version**: Must use **Python >= 3.12**.
- **Harbor Version**: Requires **Harbor >= 0.14.0, < 1.0.0**. Earlier versions cannot handle 4.0's sidecar artifacts and verifier collection hooks.
- **Docker Environment**: Docker is used by default for evaluation. Please ensure Docker Engine, Docker Compose, and the `docker` CLI are installed and available in the environment running EvalScope.
- **Network Connection**:
  - The dataset is downloaded from Harbor Hub. Please ensure network connectivity.
  - Container building and runtime may require internet access to download dependencies. Ensure a stable network environment.
- **Model Requirements**: Only the `terminus-2` agent uses your configured model for inference. Other agents (claude-code, codex, etc.) are standalone CLI tools that use their own API keys.
```

## Installation

```bash
pip install --upgrade 'evalscope[terminal_bench]'
```

## Usage

### Terminal-Bench 4.0

The default configuration loads all **66 tasks** from revision **4.0.0**. It uses Docker and `terminus-2`, inherits the official task resources and timeouts (8 hours for the agent), and uses Harbor's default turn limit (`max_turns=None`). All tasks use a separate verifier; 11 have Docker Compose services and 3 require an H100 GPU. A full run needs a Linux Docker host with H100 access, enough resources for concurrent services, and network access for dependencies.

For a local CPU smoke, explicitly select a task rather than relying on `limit` to exclude GPU or multi-container tasks:

```python
import os
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='qwen3-coder-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['terminal_bench_v4'],
    dataset_args={
        'terminal_bench_v4': {
            'extra_params': {
                'task_names': ['terminal-bench/bun-sourcemap-leak'],
            },
        },
    },
    eval_batch_size=1,
))
```

`task_names` accepts Harbor task names or glob patterns, such as `terminal-bench/ctr-*`. Filtering runs before EvalScope's shuffle, limit, repeats, and indexing. Remove `task_names` for all 66 tasks. Use `repeats=5` on `TaskConfig` for five independent trials per task; the report averages valid rewards. Subset results are debugging results, not a full benchmark score.

For Modal, install the extra and configure account credentials:

```bash
pip install --upgrade 'harbor[modal]>=0.14.0,<1.0.0'
modal token new
```

Then set `'environment_type': 'modal'` in `extra_params`. Harbor selects the task's GPU and manages separate verifier environments and Compose services. The account must have access to the requested resources.

External CLI agents can be selected with `agent_name`, for example `'claude-code'` or `'codex'`. Their inference and authentication are handled by those tools, using their own credentials; the EvalScope model API bridge is used only by `terminus-2`.
Use `eval_type='mock_llm'` to avoid creating an unused EvalScope API client, set `model` to a model supported by the selected CLI, and configure its credentials before starting EvalScope. The model name is still forwarded to Harbor; `api_url` and `api_key` do not configure the external CLI.

### Terminal-Bench 2.0 / 2.1

The following example demonstrates how to evaluate the `qwen3-coder-plus` model using evalscope.

```python
import os
from evalscope import TaskConfig, run_task

# Configure task
task_cfg = TaskConfig(
    # Model configuration
    model='qwen3-coder-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # Use OpenAI-compatible service for evaluation

    # Dataset configuration (use 'terminal_bench_v2_1' for v2.1 or 'terminal_bench_v2' for v2.0)
    datasets=['terminal_bench_v2_1'],
    dataset_args={
        'terminal_bench_v2_1': {
            'extra_params': {
                # Environment type, default is 'docker'
                # Options: 'docker', 'daytona', 'e2b', 'modal'
                'environment_type': 'docker',

                # Agent type, default is 'terminus-2'
                # Only 'terminus-2' uses your configured model for inference.
                # Other agents (claude-code, codex, etc.) run as standalone CLI tools
                # with their own API keys; the model name is forwarded to Harbor.
                'agent_name': 'terminus-2',

                # Timeout multiplier, can be increased if timeout errors occur
                'timeout_multiplier': 1.0,

                # Reproduce an absolute agent timeout. Do not combine an absolute
                # phase timeout with that phase's multiplier.
                'agent_timeout_sec': 3 * 60 * 60,
                'verifier_timeout_sec': 3 * 60 * 60,

                # Container resources for the published Qwen 3.6 protocol.
                'environment_kwargs': {
                    'override_cpus': 32,
                    'override_memory_mb': 48 * 1024,
                },

                # Maximum interaction turns, can be increased if tasks are not completed
                'max_turns': 200,
            }
        }
    },

    # Evaluation concurrency (recommended to adjust based on Docker resources, each concurrent run starts a container)
    eval_batch_size=1,

    # Limit number of evaluation samples (can be set to a small value for debugging, remove for formal evaluation)
    limit=10,
)

# Start evaluation
run_task(task_cfg)
```

## Parameter Description

The following parameters are supported in `extra_params` within `dataset_args`:

- `environment_type` (str): The environment type for running the benchmark. Default is `docker`. Supports `docker`, `daytona`, `e2b`, `modal`.
- `agent_name` (str): The agent type used in Harbor. Default is `terminus-2`. Only `terminus-2` uses the evalscope model for inference; other agents (claude-code, codex, opencode, etc.) are standalone CLI tools with their own API keys.
- `timeout_multiplier` (float): Timeout multiplier. Default is 1.0.
- `agent_timeout_sec` / `verifier_timeout_sec` (float): Final phase timeout in seconds. An absolute timeout is not
  multiplied again by `timeout_multiplier`.
- `agent_timeout_multiplier` / `verifier_timeout_multiplier` (float): Per-phase multiplier overriding the global
  multiplier. Do not set a phase multiplier together with that phase's absolute timeout.
- `max_turns` (int or None): Maximum interaction turns. Defaults to 200 for 2.0/2.1 and None (Harbor default) for 4.0.
- `task_names` (list, 4.0): Optional Harbor task names or glob patterns. Defaults to None, loading all tasks.
- `environment_kwargs` (dict): Extra kwargs passed to Harbor `EnvironmentConfig` for container resource limits. Supported keys: `override_cpus`, `override_memory_mb`, `override_storage_mb`, `override_gpus`, `force_build`, `delete`, `env`, etc.

`override_storage_mb` is only effective when the Docker storage driver and host filesystem support container quotas.
Container builds and verifier scripts may download dependencies; configure any approved mirror or proxy with
`environment_kwargs.env`. EvalScope does not infer infrastructure failure from verifier output: a verifier that writes
a valid reward of `0` is still scored as a model failure.

## Result Example

Evaluation takes a considerable amount of time, please be patient. The overall output directory structure is as follows, where model inference trials are automatically saved in the `outputs/<timestamp>/trials` directory.

```text
.
├── configs
│   └── task_config_07906d.yaml
├── logs
│   └── eval_log.log
├── predictions
│   └── qwen-plus
└── trials
    ├── adaptive-rejection-sampler__44jsCkg
    ├── bn-fit-modify__BDZ7XN8
    ├── break-filter-js-from-html__xSUatRJ
    ├── build-cython-ext__Cj6Mi7X
    ├── build-pmars__ghgk7h8
    └── build-pov-ray__JWy4tcZ
```

After evaluation completes, a result table similar to the following will be output:

```text
+------------------+-------------------+----------+----------+-------+---------+---------+
| Model            | Dataset           | Metric   | Subset   |   Num |   Score | Cat.0   |
+==================+===================+==========+==========+=======+=========+=========+
| qwen3-coder-plus | terminal_bench_v2 | Accuracy ↑ | test     |     5 |     20% | default |
+------------------+-------------------+----------+----------+-------+---------+---------+
```

## Troubleshooting

1. **Docker Connection Failed**: Please check if Docker Desktop or Docker Engine is running and the current user has permission to access the Docker socket.
2. **Docker CLI Not Found**: If EvalScope runs inside a container, mounting `/var/run/docker.sock` is not enough. The container also needs the `docker` command-line client installed.
3. **Dataset Download Failed**: The dataset is downloaded via GitHub. Please check network connectivity or configure a proxy.
4. **Python Version Error**: If you encounter syntax errors or package compatibility issues, please confirm you are using Python 3.12+.
