# AutomationBench


## Overview

AutomationBench evaluates agents on realistic business workflows across sales, marketing, operations, support,
finance, and HR. EvalScope runs the public tasks, simulated SaaS services, and assertion-based scoring provided by
Zapier's official Python package.

## Task Description

- **Task Type**: Stateful business workflows across 47 simulated SaaS tools.
- **Public Dataset**: 600 tasks, with 100 tasks in each of `sales`, `marketing`, `operations`, `support`, `finance`,
  and `hr`.
- **Simple Baseline**: The optional `simple` subset contains 200 foundational single- and two-step tasks. Its metrics
  use the `simple_` prefix and are reported separately from the public benchmark score.
- **Scoring**: `partial_credit` is the fraction of scored assertions satisfied. `pass_rate` is the mean official
  `task_completed_correctly` signal, which is 1 only when every scored assertion passes.

## Evaluation Notes

- Prepare a Python 3.13+ environment and install the pinned dependencies before evaluation:
  `python -m pip install "verifiers==0.1.12.dev2" "automation-bench @ git+https://github.com/zapier/AutomationBench.git@a321764ace3cfbe42289e6a13abef2f0f4f56fad"`.
- By default, EvalScope evaluates all six public domains (600 tasks). Use `subset_list` to select domains or `limit`
  for a smaller run.
- Use an API-backed EvalScope model with `openai_api`, `openai_responses_api`, or `anthropic_api` evaluation type.
- No Docker runtime, external dataset download, or real SaaS credentials are required. Model API credentials are still
  required.
- The released tasks are public. Zapier's official leaderboard uses a separate held-out private set, so local public
  scores are directional rather than leaderboard-identical.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `automation_bench` |
| **Dataset ID** | [AutomationBench](https://github.com/zapier/AutomationBench) |
| **Paper** | [Paper](https://arxiv.org/abs/2604.18934) |
| **Tags** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **Metrics** | `pass_rate`, `partial_credit`, `error_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 600 |
| Prompt Length (Mean) | 983.57 chars |
| Prompt Length (Min/Max) | 586 / 1785 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `sales` | 100 | 832.89 | 586 | 1245 |
| `marketing` | 100 | 953.02 | 612 | 1275 |
| `operations` | 100 | 1259.54 | 787 | 1785 |
| `support` | 100 | 937.56 | 711 | 1577 |
| `finance` | 100 | 990.51 | 709 | 1259 |
| `hr` | 100 | 927.91 | 689 | 1276 |

## Sample Example

**Subset**: `sales`

```json
{
  "input": [
    {
      "id": "97a20b04",
      "content": "You are a workflow automation agent. Execute the requested tasks using the available tools. Do not ask clarifying questions - use the information provided and make reasonable assumptions when needed. You have a budget of ~50 tool-using turns  ... [TRUNCATED 47 chars] ...  searches. When summarizing your work in messages or records, list only items you acted on. Do not name, enumerate, or explain items you skipped, excluded, or rejected — handle exclusions silently in the action, not narratively in the output.",
      "source": "input",
      "role": "system"
    },
    {
      "id": "4d75f740",
      "content": "We just closed the Meridian Corp Platform Deal! Mark it as won and route the win notice to the right team per our routing policy. Be sure to follow the latest routing guidelines. Confirm the account tier from the 'Account Hierarchy' spreadshe ... [TRUNCATED 160 chars] ... support-escalation@example.com, executive-team@example.com, sales-team@example.com, smb-team@example.com, vp-sales@example.com\n\nUse Gmail for all email sends. Include the names of affected entities and the relevant amounts in your message(s).",
      "source": "input",
      "role": "user"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "sales",
  "metadata": {
    "record_key": "sales:0",
    "domain": "sales",
    "task": "sales.multi_hop_lookup",
    "official_example_id": 501
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `toolset` | `str` | `api` | Official tool style exposed to the agent. Choices: ['api', 'zapier', 'limited_zapier'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets automation_bench \
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
    datasets=['automation_bench'],
    dataset_args={
        'automation_bench': {
            # subset_list: ['sales', 'marketing', 'operations']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
