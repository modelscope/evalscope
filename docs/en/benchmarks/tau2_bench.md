# τ²-bench


## Overview

τ²-bench (Tau Squared Bench) is an extension and enhancement of the original τ-bench. It evaluates conversational AI agents in domain-specific scenarios with expanded capabilities including telecom domain support.

## Task Description

- **Task Type**: Advanced Conversational Agent Evaluation
- **Input**: User scenarios with complex goals and multi-step requirements
- **Output**: Agent actions via API tool calls following policy guidelines
- **Domains**: Airline, Retail, Telecom

## Key Features

- Extended domain coverage (adds telecom to airline and retail)
- Enhanced task complexity and evaluation criteria
- LLM-simulated user interactions
- Multi-turn dialogue with tool calling
- Policy compliance verification

## Evaluation Notes

- **Installation Required**: `pip install git+https://github.com/sierra-research/tau2-bench@v0.2.0`
- **User Model Configuration**: Requires setting up a user simulation model
- Primary metric: **Accuracy** based on task completion reward
- Supports **airline**, **retail**, and **telecom** domains
- Uses **pass@k** aggregation for robustness evaluation
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau2_bench.html)
- A newer release **τ³-bench** (v1.0.0) is available as the `tau3_bench` benchmark, with a new `banking_knowledge` domain and 75+ task fixes. Note: the two versions cannot be installed in the same environment.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tau2_bench` |
| **Dataset ID** | [evalscope/tau2-bench-data](https://modelscope.cn/datasets/evalscope/tau2-bench-data/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling`, `Reasoning` |
| **Metrics** | N/A |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_hat_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 278 |
| Prompt Length (Mean) | 46.65 chars |
| Prompt Length (Min/Max) | 0 / 658 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `airline` | 50 | 127.94 | 29 | 658 |
| `retail` | 114 | 1.95 | 0 | 145 |
| `telecom` | 114 | 55.7 | 51 | 59 |

## Sample Example

**Subset**: `airline`

```json
{
  "input": [
    {
      "id": "27580cd1",
      "content": "Testing that agent refuses to proceed with a cancellation that is not allowed even if User mentions that she had been told she didn't need insurance."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "airline",
  "metadata": {
    "id": "0",
    "description": {
      "purpose": "Testing that agent refuses to proceed with a cancellation that is not allowed even if User mentions that she had been told she didn't need insurance.",
      "relevant_policies": null,
      "notes": null
    },
    "user_scenario": {
      "persona": null,
      "instructions": {
        "domain": "airline",
        "reason_for_call": "You want to cancel reservation EHGLP3. \n\nIt may be more than 24 hours after booking, but it is ok because you were out of town for that time.",
        "known_info": "You are Emma Kim.\nYour user id is emma_kim_9957.",
        "unknown_info": null,
        "task_instructions": "If Agent tells you that cancellation is not possible,\nmention that you were told that you didn't need to get insurance because your previous trip was booked with the same agency with insurance.\n\nYou don't want to cancel if you don't get a refund."
      }
    },
    "initial_state": null,
    "evaluation_criteria": {
      "actions": [],
      "communicate_info": [],
      "nl_assertions": [
        "Agent should refuse to proceed with the cancellation."
      ]
    }
  }
}
```

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
    --datasets tau2_bench \
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
    datasets=['tau2_bench'],
    dataset_args={
        'tau2_bench': {
            # subset_list: ['airline', 'retail', 'telecom']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
