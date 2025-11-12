# τ-bench

## Introduction

τ-bench (TAU Bench) is a specialized benchmark framework designed to evaluate the tool utilization capabilities of large language models in **dynamic dialogue agent scenarios**. This benchmark simulates dynamic conversations between a user (simulated by a language model) and a language agent, where the agent is equipped with domain-specific API tools and strategic guidance.

- **Project URL**: [https://github.com/sierra-research/tau-bench](https://github.com/sierra-research/tau-bench)

Key features of this benchmark include:

- **Dynamic Interaction**: Simulates multi-turn dialogues between real users and AI agents.
- **Tool Integration**: Agents must effectively utilize provided API tools.
- **Strategy Compliance**: Agents must adhere to specific business strategies and guidelines.
- **Domain Expertise**: Covers real-world business scenarios such as airline customer service and retail customer service.

Supported evaluation domains:
- `airline`: Airline customer service scenarios
- `retail`: Retail customer service scenarios

## Dependency Installation

Before running the evaluation, install the following dependencies:

```bash
pip install evalscope  # Install evalscope
pip install git+https://github.com/sierra-research/tau-bench  # Install tau-bench
```

## Usage

Run the following code to start the evaluation. The example below uses the qwen-plus model for evaluation.

**⚠️ Note: Only API model service evaluation is supported. For local model evaluation, it is recommended to use frameworks like vLLM to pre-launch the service. The official default user model is gpt-4o, and to achieve leaderboard results, please use this model for evaluation.**

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # Use API model service
    datasets=['tau_bench'],
    dataset_args={
        'tau_bench': {
            'subset_list': ['airline', 'retail'],  # Select evaluation domains
            'extra_params': {
                'user_model': 'qwen-plus',  # User simulation model
                'api_key': os.getenv('DASHSCOPE_API_KEY'),
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'generation_config': {
                    'temperature': 0.7,
                }
            }
        }
    },
    eval_batch_size=5,
    limit=5,  # Limit the number of evaluations for quick testing; remove for full evaluation
    generation_config={
        'temperature': 0.6,
    },
)

run_task(task_cfg=task_cfg)
```

## Evaluation Methodology

### Dialogue Process

The evaluation process of τ-bench includes the following key steps:

1. **Task Initialization**: The system provides the agent with domain-specific API toolsets and strategic guidance.
2. **User Simulation**: The user simulator generates natural user requests based on predefined scenarios.
3. **Agent Response**: The tested agent model generates responses based on tools and strategies.
4. **Multi-turn Interaction**: The user and agent engage in multi-turn dialogues until the task is completed or fails.
5. **Result Evaluation**: Scoring is based on task completion and strategy adherence.

### Evaluation Dimensions

- Whether the agent successfully completes the core task of the user request.
- Whether the necessary API tools are used correctly.

### Domain Characteristics

#### Airline Scenario
- **API Tools**: Flight inquiry, booking modification, seat selection, ticket changes, etc.
- **Strategy Requirements**: Fee transparency, customer prioritization, adherence to safety regulations.
- **Typical Tasks**: Flight rescheduling, seat upgrades, baggage issue handling.

#### Retail Scenario
- **API Tools**: Product inquiry, order management, inventory check, payment processing, etc.
- **Strategy Requirements**: Price accuracy, real-time inventory, customer satisfaction.
- **Typical Tasks**: Product recommendations, order tracking, return and exchange processing.

## Result Analysis

### Output Example

```text
+-----------+-----------+--------+---------+-------+---------+
| Model     | Dataset   | Metric | Subset  |   Num |   Score |
+===========+===========+========+=========+=======+=========+
| qwen-plus | tau_bench | Pass^1 | airline |    10 |   0.5   |
+-----------+-----------+--------+---------+-------+---------+
| qwen-plus | tau_bench | Pass^1 | retail  |    10 |   0.6   |
+-----------+-----------+--------+---------+-------+---------+
| qwen-plus | tau_bench | Pass^1 | OVERALL |    20 |   0.55  |
+-----------+-----------+--------+---------+-------+---------+
```

### Performance Analysis Metrics

**Pass^1**: The proportion of tasks successfully completed on the first attempt
- Measures the model's ability to resolve user issues in a single dialogue.
- Considers task completion, strategy adherence, and dialogue quality comprehensively.