# Toolathlon Official Service Wrapper


## Overview

Toolathlon is an agent benchmark for realistic, long-horizon tool use across many MCP-backed software
environments. This EvalScope benchmark is a wrapper around the official Toolathlon remote evaluation service,
not a local reimplementation of the MCP environments or official evaluator.

## Evaluation Mode

- Benchmark id: `toolathlon`
- Supported mode: official service private mode
- EvalScope controls model endpoint, task selection, job parameters, polling, result download, and reporting
- The official Toolathlon service controls MCP environments, task containers, agent loop execution, and scoring
- No Toolathlon, MCP application accounts, or Toolathlon Python package installation are required when using the
  official public evaluation service
- A local or intranet OpenAI-compatible endpoint is required for private mode
- EvalScope represents one remote Toolathlon job as one local sample; the generated data statistics count wrapper
  jobs, while `task_list` and `limit` control the Toolathlon tasks submitted inside that job
- The bundled Toolathlon-Verified task list was inspected from official repository commit
  `b7bbac3f9a1f381b095c878debe1a47dd164ad85`

## Usage Guide

See the Toolathlon usage guide for public-service limits, private-mode data flow, self-hosted service setup, and
EvalScope configuration examples:

- https://evalscope.readthedocs.io/en/latest/third_party/toolathlon.html

Official sources:

- https://github.com/hkust-nlp/Toolathlon
- https://github.com/hkust-nlp/Toolathlon/blob/main/EVAL_SERVICE_README.md


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `toolathlon` |
| **Dataset ID** | [Toolathlon](https://github.com/hkust-nlp/Toolathlon) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1 |
| Prompt Length (Mean) | 50 chars |
| Prompt Length (Min/Max) | 50 / 50 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "22a512d8",
      "content": "Run Toolathlon official remote evaluation service."
    }
  ],
  "target": "",
  "id": 0,
  "metadata": {
    "task_list": [
      "ab-testing",
      "academic-pdf-report",
      "academic-warning",
      "add-bibtex",
      "apply-phd-email",
      "arrange-workspace",
      "canvas-arrange-exam",
      "canvas-art-manager",
      "canvas-art-quiz",
      "canvas-do-quiz",
      "... [TRUNCATED 98 more items] ..."
    ],
    "mode": "private"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `private` | Toolathlon service mode. EvalScope supports private mode for this wrapper. Choices: ['private'] |
| `server_host` | `str` | `47.253.6.47` | Official Toolathlon evaluation service host. |
| `server_port` | `int` | `8080` | Official Toolathlon HTTP service port. |
| `ws_proxy_port` | `int` | `8081` | Official Toolathlon WebSocket proxy port for private mode. |
| `workers` | `int` | `10` | Number of parallel Toolathlon workers requested from the official service. |
| `provider` | `str` | `unified` | Toolathlon model provider type. Choices: ['unified', 'openai_stateful_responses'] |
| `task_list` | `list` | `[]` | Optional Toolathlon task names to evaluate. Empty uses the bundled Toolathlon-Verified list. |
| `task_list_file` | `str` | `` | Optional file containing one Toolathlon task name per line. |
| `model_params` | `dict` | `{}` | Extra model parameters forwarded to Toolathlon, merged after TaskConfig.generation_config. |
| `job_id` | `str` | `` | Optional Toolathlon job id. Reuse to resume an incomplete official-service job. |
| `force_redownload` | `bool` | `False` | Force redownload of Toolathlon result archives. |
| `override_output_dir` | `bool` | `False` | Clear the Toolathlon output directory when it already contains files. |
| `skip_container_restart` | `bool` | `False` | Skip Toolathlon container restart. Use only for small debug task subsets. |
| `trust_env_in_httpx` | `bool` | `False` | Allow httpx to use proxy environment variables. |
| `timeout_seconds` | `int` | `14400` | Maximum time to wait for a Toolathlon official-service job. |
| `poll_interval` | `int` | `5` | Polling interval in seconds for Toolathlon job status. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets toolathlon \
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
    datasets=['toolathlon'],
    dataset_args={
        'toolathlon': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
