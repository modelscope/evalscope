# Terminal-Bench 2.0

## Introduction

Terminal-Bench v2 is a command-line benchmark suite designed to evaluate AI agents on 89 real-world multi-step terminal tasks. These tasks cover a wide range of scenarios from compilation and debugging to system administration, all executed in isolated containers. Each task is rigorously validated and automatically scored (0/1), aiming to push frontier models to prove they can not only answer questions but also take action.

EvalScope integrates the evaluation functionality of Terminal-Bench v2, allowing users to conveniently evaluate models that support API services using the EvalScope framework.

- Project Repository: [harbor](https://github.com/laude-institute/harbor)
- Dataset Repository: [terminal-bench-2](https://github.com/laude-institute/terminal-bench-2)

## Requirements

```{important}
- **Python Version**: Must use **Python >= 3.12**.
- **Docker Environment**: Docker is used by default for evaluation. Please ensure Docker Engine and Docker Compose are installed and running locally.
- **Network Connection**:
  - The dataset will be downloaded directly from GitHub. Please ensure network connectivity.
  - Container building and runtime may require internet access to download dependencies. Ensure a stable network environment.
- **Model Requirements**: Only supports evaluation of models via API services (it is recommended to expose local models as services using frameworks like vLLM).
```

## Installation

```bash
pip install evalscope
# Install harbor library (core dependency for Terminal-Bench)
pip install harbor==0.1.28
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

    # Dataset configuration
    datasets=['terminal_bench_v2'],
    dataset_args={
        'terminal_bench_v2': {
            'extra_params': {
                # Environment type, default is 'docker'
                # Options: 'docker', 'daytona', 'e2b', 'modal'
                'environment_type': 'docker',
                
                # Agent type, default is 'terminus-2'
                # Options: 'oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'
                # 'oracle' is the ground truth agent, can be used to test if environment is configured correctly
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
- `agent_name` (str): The agent type used in Harbor. Default is `terminus-2`.
- `timeout_multiplier` (float): Timeout multiplier. Default is 1.0.
- `max_turns` (int): Maximum interaction turns for the agent to complete tasks. Default is 200.

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
2. **Dataset Download Failed**: The dataset is downloaded via GitHub. Please check network connectivity or configure a proxy.
3. **Python Version Error**: If you encounter syntax errors or package compatibility issues, please confirm you are using Python 3.12+.