# τ-bench


## Overview

τ-bench (Tau Bench) is a benchmark for evaluating conversational AI agents that interact with users through domain-specific API tools and policy guidelines. It simulates dynamic, multi-turn conversations where a language model acts as both the user and the agent.

## Task Description

- **Task Type**: Conversational Agent Evaluation
- **Input**: User scenarios with specific goals and constraints
- **Output**: Agent actions via API tool calls to complete tasks
- **Domains**: Airline customer service, Retail customer service

## Key Features

- Dynamic conversation simulation with LLM-simulated users
- Domain-specific API tools and policy guidelines
- Realistic customer service scenarios
- Tests multi-turn dialogue capabilities
- Evaluates tool use and policy compliance

## Evaluation Notes

- **Installation Required**: `pip install git+https://github.com/sierra-research/tau-bench`
- **User Model Configuration**: Requires setting up a user simulation model
- Primary metric: **Accuracy** based on task completion reward
- Supports **airline** and **retail** domains
- Uses **pass@k** aggregation for robustness evaluation
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau_bench.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tau_bench` |
| **Dataset ID** | [tau-bench](https://github.com/sierra-research/tau-bench) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling`, `Reasoning` |
| **Metrics** | N/A |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_hat_k` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_model` | `str` | `qwen-plus` | Model used to simulate the user in the environment. |
| `api_key` | `str` | `EMPTY` | API key for the user model backend. |
| `api_base` | `str` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Base URL for the user model API requests. |
| `generation_config` | `dict` | `{'temperature': 0.0}` | Default generation config for user model simulation. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tau_bench \
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
    datasets=['tau_bench'],
    dataset_args={
        'tau_bench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


