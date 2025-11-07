# τ²-bench

## Introduction

τ²-bench (Tau Squared Bench) is an extended and enhanced version of τ-bench, incorporating a series of code fixes and adding new telecom domain troubleshooting scenarios. It is designed to evaluate large language models' tool usage and policy adherence capabilities in dynamic conversational agents.

- Project URL: https://github.com/sierra-research/tau2-bench
- Important Note: τ²-bench is the latest version, and it is recommended to prioritize using τ²-bench for evaluation.

Core Features:
- Dynamic Interaction: Simulates multi-turn conversations between real users and AI agents
- Tool Integration: Agents need to appropriately use provided API tools
- Policy Adherence: Agents need to follow business policies and guidelines
- Domain Extension: Adds telecom troubleshooting scenarios on top of airline and retail foundations
- Reliability Improvement: Code fixes and stability improvements based on τ-bench

Supported Evaluation Domains:
- airline: Airline customer service
- retail: Retail customer service
- telecom: Telecom customer service (newly added, including network/plan/billing/troubleshooting, etc.)

## Installation

```bash
pip install evalscope
# Install tau2-bench
pip install "git+https://github.com/sierra-research/tau2-bench@v0.2.0"
```

```{important}
- The dataset is automatically pulled from ModelScope by evalscope (Dataset ID: evalscope/tau2-bench-data), and TAU2_DATA_DIR is automatically configured.
- Only supports evaluating the tested model through API services (it is recommended to expose local models as services using frameworks like vLLM first).
```

## Usage

Taking qwen-plus as an example. The official leaderboard typically uses `user model = gpt-4.1-2025-04-14`. To align with the leaderboard, configure user_model as gpt-4.1-2025-04-14 and provide the corresponding API Key and Base URL.

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    # Agent Model under test
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # Evaluate using OpenAI-compatible service

    datasets=['tau2_bench'],
    dataset_args={
        'tau2_bench': {
            'subset_list': ['airline', 'retail', 'telecom'],  # Supports three domains
            'extra_params': {
                # User Model for simulating user behavior and driving conversation environment
                'user_model': 'qwen-plus',  # Can change to 'gpt-4.1-2025-04-14' to align with official leaderboard
                'api_key': os.getenv('DASHSCOPE_API_KEY'),
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'generation_config': {
                    'temperature': 0.7,
                }
            }
        }
    },

    eval_batch_size=5,  # Evaluation concurrency size
    limit=5,  # Recommended for quick trial runs, can be removed for formal evaluation
    generation_config={
        'temperature': 0.6,
    },
)

run_task(task_cfg)
```

Tips:
- If using gpt-4.1-2025-04-14 as the user simulation model, please configure:
  - extra_params.user_model='gpt-4.1-2025-04-14'
  - extra_params.api_base='https://api.openai.com/v1'
  - extra_params.api_key=<OPENAI_API_KEY>

## Evaluation Process

1) Task Initialization: Provide the agent with domain API tools and policy guidelines
2) User Simulation: User model generates natural requests according to scenarios
3) Agent Response: Tested model generates responses based on tools and policies
4) Multi-turn Interaction: Continuous conversation until task completion or failure
5) Result Evaluation: Scoring based on task completion and policy adherence

## Evaluation Dimensions

- Whether user goals are achieved (task completion rate)
- Whether necessary API tools are correctly invoked
- Whether business policies and constraints are followed

## Domain Characteristics

- Airline
  - Tools: Flight query, rebooking, seating, refund/change, etc.
  - Typical Tasks: Rebooking, seat upgrade, baggage issue handling
- Retail
  - Tools: Product/order/inventory/payment, etc.
  - Typical Tasks: Product recommendation, order tracking, return/exchange handling
- Telecom (Newly Added)
  - Tools: Network diagnosis, plan changes, service suspension/restoration, trouble tickets, etc.
  - Typical Tasks: Network connectivity issues, billing disputes, plan upgrades and troubleshooting

## Example Results

```text
+-----------+------------+-------------+----------+-------+---------+---------+
| Model     | Dataset    | Metric      | Subset   |   Num |   Score | Cat.0   |
+===========+============+=============+==========+=======+=========+=========+
| qwen-plus | tau2_bench | mean_Pass^1 | airline  |    10 |     0.6 | default |
+-----------+------------+-------------+----------+-------+---------+---------+
| qwen-plus | tau2_bench | mean_Pass^1 | retail   |    10 |     0.7 | default |
+-----------+------------+-------------+----------+-------+---------+---------+
| qwen-plus | tau2_bench | mean_Pass^1 | telecom  |    10 |     0.8 | default |
+-----------+------------+-------------+----------+-------+---------+---------+
| qwen-plus | tau2_bench | mean_Pass^1 | OVERALL  |    30 |     0.7 | -       |
+-----------+------------+-------------+----------+-------+---------+---------+ 
```

## Metric Description

- Pass^1: The proportion of tasks completed on the first attempt (higher is better)
  - Reflects the correctness of tool usage, policy adherence, and goal achievement within a single conversation