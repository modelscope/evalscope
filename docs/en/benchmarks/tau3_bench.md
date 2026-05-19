# τ³-bench


## Overview

τ³-bench (Tau Cubed Bench) is the v1.0.0 release of the tau-bench family. It extends τ²-bench with a knowledge-retrieval domain, voice/audio-native evaluation, and 75+ task fixes across the existing domains.

## Task Description

- **Task Type**: Conversational Agent Evaluation with optional knowledge retrieval
- **Input**: User scenarios with complex goals and multi-step requirements
- **Output**: Agent actions via API tool calls following policy guidelines
- **Domains**: Airline, Retail, Telecom, Banking Knowledge

## Key Features

- New `banking_knowledge` domain with 97 tasks and 698 policy/procedure documents (RAG)
- 75+ task quality fixes across airline / retail / banking domains
- Pluggable retrieval pipeline: BM25, dense embeddings (OpenAI / Qwen), grep, sandbox shell, rerankers
- LLM-simulated user interactions, multi-turn dialogue with tool calling

## Evaluation Notes

- **Python**: requires 3.12-3.13
- **Installation Required**: `pip install 'tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0'`
- **Cannot coexist with `tau2_bench`** in the same environment (same PyPI package name `tau2`, different versions). Pick one.
- **User Model Configuration**: Requires setting up a user simulation model
- **Retrieval config (banking_knowledge only)**: defaults to `bm25` (offline). Switch via `extra_params.retrieval_config`. Other configs may need extra deps:
  - `bm25` → ships with `[knowledge]` extra (no API key)
  - `openai_embeddings*` → set `OPENAI_API_KEY`
  - `qwen_embeddings*` → set `OPENROUTER_API_KEY`
  - `*_reranker` → also needs `OPENAI_API_KEY`
  - `terminal_use` / `alltools*` → require Anthropic `sandbox-runtime` (npm) + ripgrep / bwrap / socat (see tau2 README)
- Primary metric: **Accuracy** based on task completion reward
- Uses **pass@k** aggregation for robustness evaluation
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau3_bench.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tau3_bench` |
| **Dataset ID** | [evalscope/tau3-bench-data](https://modelscope.cn/datasets/evalscope/tau3-bench-data/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling`, `Reasoning` |
| **Metrics** | N/A |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `mean_and_pass_hat_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 375 |
| Prompt Length (Mean) | 39.22 chars |
| Prompt Length (Min/Max) | 0 / 661 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `airline` | 50 | 135.58 | 29 | 661 |
| `retail` | 114 | 1.95 | 0 | 145 |
| `telecom` | 114 | 55.7 | 51 | 59 |
| `banking_knowledge` | 97 | 14 | 14 | 14 |

## Sample Example

**Subset**: `airline`

```json
{
  "input": [
    {
      "id": "333afe68",
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
      ],
      "reward_basis": [
        "DB",
        "COMMUNICATE"
      ]
    },
    "_domain": "airline"
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
| `retrieval_config` | `str` | `bm25` | Retrieval config name for the banking_knowledge domain. Common values: no_knowledge, full_kb, golden_retrieval, bm25, openai_embeddings, qwen_embeddings, *_reranker, *_grep, terminal_use, alltools. Ignored for non-knowledge domains. |
| `retrieval_config_kwargs` | `dict` | `{}` | Optional kwargs forwarded to the retrieval pipeline. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tau3_bench \
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
    datasets=['tau3_bench'],
    dataset_args={
        'tau3_bench': {
            # subset_list: ['airline', 'retail', 'telecom']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


