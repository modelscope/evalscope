# Terminal-Bench

## Introduction

Terminal-Bench is a command-line benchmark suite designed to evaluate AI agents on real-world multi-step terminal tasks. These tasks cover a wide range of scenarios from compilation and debugging to system administration, all executed in isolated containers. Each task is rigorously validated and automatically scored (0/1), aiming to push frontier models to prove they can not only answer questions but also take action.

EvalScope supports two versions:

- **`terminal_bench_v2`**: The original 89-task benchmark (Terminal-Bench 2.0).
- **`terminal_bench_v2_1`**: An improved iteration with 26 task fixes addressing bugs, timeout adjustments, and reward hacking prevention (Terminal-Bench 2.1, recommended).

Links:

- Project Repository: [harbor](https://github.com/laude-institute/harbor)
- Dataset (Hub): [terminal-bench-2](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2/latest) | [terminal-bench-2-1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest)

## Requirements

```{important}
- **Python Version**: Must use **Python >= 3.12**.
- **Docker Environment**: Docker is used by default for evaluation. Please ensure Docker Engine, Docker Compose, and the `docker` CLI are installed and available in the environment running EvalScope.
- **Network Connection**:
  - The dataset is downloaded from Harbor Hub. Please ensure network connectivity.
  - Container building and runtime may require internet access to download dependencies. Ensure a stable network environment.
- **Model Requirements**: Only the `terminus-2` agent uses your configured model for inference. Other agents (claude-code, codex, etc.) are standalone CLI tools that use their own API keys.
```

## Installation

```bash
pip install evalscope[terminal_bench]
```

## Usage

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
                # with their own API keys and ignore the model configuration.
                'agent_name': 'terminus-2',

                # Timeout multiplier, can be increased if timeout errors occur
                'timeout_multiplier': 1.0,

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
- `max_turns` (int): Maximum interaction turns for the agent to complete tasks. Default is 200.
- `environment_kwargs` (dict): Extra kwargs passed to Harbor `EnvironmentConfig` for container resource limits. Supported keys: `override_cpus`, `override_memory_mb`, `override_storage_mb`, `override_gpus`, `force_build`, `delete`, `env`, etc.

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
| qwen3-coder-plus | terminal_bench_v2 | mean_acc | test     |     5 |     0.2 | default |
+------------------+-------------------+----------+----------+-------+---------+---------+
```

## Troubleshooting

1. **Docker Connection Failed**: Please check if Docker Desktop or Docker Engine is running and the current user has permission to access the Docker socket.
2. **Docker CLI Not Found**: If EvalScope runs inside a container, mounting `/var/run/docker.sock` is not enough. The container also needs the `docker` command-line client installed.
3. **Dataset Download Failed**: The dataset is downloaded via GitHub. Please check network connectivity or configure a proxy.
4. **Python Version Error**: If you encounter syntax errors or package compatibility issues, please confirm you are using Python 3.12+.
