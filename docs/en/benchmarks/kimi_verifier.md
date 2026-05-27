# Kimi-Vendor-Verifier (Param Compliance)


## Overview

Kimi-Vendor-Verifier is a pre-flight compliance check for Kimi K2 / K2-Thinking deployments. It sends synthetic probe requests to verify that the vendor API correctly **rejects** non-default values of immutable decoding parameters (``temperature``, ``top_p``, ``presence_penalty``, ``frequency_penalty``, ``n``) and **accepts** their defaults. A vendor that silently accepts wrong values risks producing degraded model output that does not match official Moonshot AI behavior. Adapted from [Kimi-Vendor-Verifier/verify_params.py](https://github.com/MoonshotAI/Kimi-Vendor-Verifier/blob/main/verify_params.py).

## Task Description

- **Task Type**: API parameter-compliance probing (deployment health check)
- **Input**: A minimal chat message plus a single test parameter and thinking-mode `extra_body`
- **Output**: Whether the vendor accepted (HTTP 200) or rejected (HTTP 400) the request
- **Dataset**: Fully synthetic — no external dataset is downloaded; probes are generated in code from the K2 spec

## Key Features

- Synthetic probe set: one ``no_param`` sanity probe + 5 default-value (accept) probes + 5 wrong-value (reject) probes per (subset × thinking) combination
- Three subsets covering all common Kimi deployment shapes:
    - ``kimi`` — official Moonshot SaaS API (``extra_body = {"thinking": {"type": ...}}``); thinking on/off
    - ``opensource`` — vLLM / SGLang / KTransformers chat-template hook (``extra_body = {"chat_template_kwargs": {"thinking": ...}}``); thinking on/off
    - ``none`` — non-hybrid model; no thinking parameter sent
- HTTP 400 responses are treated as the success signal when a reject was expected
- Single small request per probe; total cost is negligible compared to a full benchmark

## Evaluation Notes

- Default configuration uses **0-shot** synthetic probes
- Metrics: **param_immutable_reject_rate**, **param_default_accept_rate**, **inference_error_rate**
- Only HTTP 400 (``BadRequestError``) counts as a real parameter rejection; transport errors (5xx / timeout / 429) are excluded from the reject/accept denominators and surfaced via ``inference_error_rate`` so a flaky vendor doesn't get a free pass
- A correctly-deployed Kimi K2 vendor should report both rate metrics at **1.0** with ``inference_error_rate = 0``; anything less indicates a parameter-enforcement gap or transport instability
- For non-Kimi models, expect ``param_immutable_reject_rate = 0`` (no K2 spec to enforce) and ``param_default_accept_rate = 1.0`` (sensible defaults accepted)
- Select subset via ``dataset_args={'kimi_verifier': {'subset_list': ['kimi']}}`` (or ``opensource`` / ``none``)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `kimi_verifier` |
| **Dataset ID** | `kimi_verifier` |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling` |
| **Metrics** | `param_immutable_reject_rate`, `param_default_accept_rate`, `inference_error_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 55 |
| Prompt Length (Mean) | 26 chars |
| Prompt Length (Min/Max) | 26 / 26 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `kimi` | 22 | 26 | 26 | 26 |
| `opensource` | 22 | 26 | 26 | 26 |
| `none` | 11 | 26 | 26 | 26 |

## Sample Example

**Subset**: `kimi`

```json
{
  "input": [
    {
      "id": "03c069db",
      "content": "Say 'OK' and nothing else."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "kimi",
  "metadata": {
    "think_mode": "kimi",
    "thinking": false,
    "param_name": null,
    "test_value": null,
    "expected_reject": false
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
    --datasets kimi_verifier \
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
    datasets=['kimi_verifier'],
    dataset_args={
        'kimi_verifier': {
            # subset_list: ['kimi', 'opensource', 'none']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


