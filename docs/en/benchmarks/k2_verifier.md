# K2-Vendor-Verifier


## Overview

K2-Vendor-Verifier checks whether a third-party deployment of Kimi-K2 faithfully reproduces the official Moonshot AI API's tool-calling behavior. It replays the official evaluation prompt set against a vendor endpoint and compares finish_reason and tool-call payloads against the official baseline. Adapted from [MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier).

## Task Description

- **Task Type**: Vendor-deployment correctness check (tool calling)
- **Input**: Multi-turn chat messages with available tool definitions, identical to the upstream K2VV prompt set
- **Output**: Vendor's chat-completion response (finish_reason and tool_calls)
- **Comparison**: Vendor's behavior is compared against the official Moonshot AI baseline shipped in the dataset

## Key Features

- Uses the official 2,000-row K2-Thinking sample set (50% of the upstream test set)
- Reports the K2VV primary metric `trigger_similarity` — F1 of the tool-call decision against the official baseline
- Schema-validates triggered tool-call arguments against the declared JSON schema
- Surfaces raw counts for sanity checks (`count_finish_reason_tool_calls`, `count_successful_tool_call`)
- Hosted dataset preserves official `finish_reason` and `tool_calls` so future metrics can compare payload-level fidelity

## Evaluation Notes

- Default configuration uses **0-shot** evaluation; multi-turn context is part of each sample
- Metrics: **trigger_similarity**, **schema_accuracy**, **count_finish_reason_tool_calls**, **count_successful_tool_call**
- A `trigger_similarity` ≥ 0.73 against the official baseline is the rough acceptance threshold per the upstream K2VV README
- Only the `k2_thinking` subset is published (K2-0905 to follow when upstream releases it)
- A few historical assistant messages in the upstream baseline have malformed JSON in `tool_calls.arguments`; the adapter sanitizes them on load


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `k2_verifier` |
| **Dataset ID** | [evalscope/K2VendorVerifier](https://modelscope.cn/datasets/evalscope/K2VendorVerifier/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling` |
| **Metrics** | `trigger_similarity`, `schema_accuracy`, `count_finish_reason_tool_calls`, `count_successful_tool_call` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `f1` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets k2_verifier \
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
    datasets=['k2_verifier'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
