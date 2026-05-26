# MiniMax-Vendor-Verifier


## Overview

MiniMax-Vendor-Verifier is a multi-validator deployment-correctness check for MiniMax M2 / M2.5 / M2.7 vendors. Each prompt row carries an optional ``check_type`` tag that routes it through specific validators, plus an always-on ``error_only_reasoning`` detector for the most common deployment regression. Adapted from [MiniMax-Provider-Verifier](https://github.com/MiniMax-AI/MiniMax-Provider-Verifier).

## Task Description

- **Task Type**: Vendor-deployment correctness check (multi-dimensional)
- **Input**: Multi-turn chat messages with optional tool definitions, plus per-row routing tags (``check_type``, ``expected_tool_call``)
- **Output**: Vendor's chat-completion response, scored against the validator(s) selected for that row
- **Dispatch**: Rows without ``check_type`` default to the ``tool_calls`` validator; rows with ``check_type`` run only the listed validators

## Key Features

- Five upstream validators ported as pure functions:
    - ``tool_calls`` — JSON-schema validation of arguments + array-command soundness check, plus a confusion matrix over ``expected_tool_call``
    - ``error_only_reasoning`` (always-on) — flags responses with reasoning but no content and no tool calls (a deployment regression)
    - ``contains_russian_characters_unicode`` — language-following check; fails when Cyrillic codepoints leak into the response
    - ``repeat_n_gram`` — degenerate-repetition detector (any 3-gram appearing 4 or more times)
    - ``scenario_check`` — verifies the model preserves the declared JSON property order, catching providers that re-sort ``parameters.properties``
- Per-validator denominator in the report: ``num=0`` indicates no row in the subset triggered that validator (not a failure)
- Hosted dataset preserves the upstream sample.jsonl plus per-loop baseline traces for M2.5 / M2.7

## Evaluation Notes

- Default configuration uses **0-shot** evaluation; the ``default`` subset has 102 rows
- Metrics: **tool_calls_match_rate**, **schema_accuracy**, **error_only_reasoning_rate**, **language_following_success_rate**, **repeat_ngram_pass_rate**, **scenario_check_pass_rate**
- Per upstream guidance, a correctly-deployed vendor should hit ``tool_calls_match_rate ≈ 0.98``, ``schema_accuracy ≥ 0.98``, ``error_only_reasoning_rate = 0``, and ``scenario_check_pass_rate = 1.0``
- When using ``--limit``, the rarer ``check_type`` rows (scenario / repeat / language) may not all be sampled; check the per-validator ``num`` column


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `minimax_verifier` |
| **Dataset ID** | [evalscope/MiniMaxVendorVerifier](https://modelscope.cn/datasets/evalscope/MiniMaxVendorVerifier/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling` |
| **Metrics** | `tool_calls_match_rate`, `schema_accuracy`, `error_only_reasoning_rate`, `language_following_success_rate`, `repeat_ngram_pass_rate`, `scenario_check_pass_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 102 |
| Prompt Length (Mean) | 72251.38 chars |
| Prompt Length (Min/Max) | 16 / 341252 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "6cc50a79",
      "content": "日本ではどのような時にお年玉を渡しますか？",
      "role": "user"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "tools": [],
  "metadata": {
    "check_type": [
      "contains_russian_characters_unicode"
    ],
    "expected_tool_call": null,
    "tools_raw": []
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets minimax_verifier \
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
    datasets=['minimax_verifier'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


