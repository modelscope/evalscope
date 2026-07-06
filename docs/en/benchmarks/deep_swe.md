# DeepSWE


## Overview

DeepSWE is a coding-agent benchmark for evaluating repository-level software engineering tasks. EvalScope
integrates it through Pier and runs each benchmark sample as one Pier Python API job.

## Task Description

- **Task Type**: Agentic software engineering
- **Input**: DeepSWE task directory containing task metadata and verifier assets
- **Output**: A repository patch produced by a Pier built-in agent
- **Scoring**: Binary verifier reward exposed as `acc`

## Evaluation Notes

- Requires **Python>=3.12**, Docker, and `pip install evalscope[deep_swe]`
- Dataset defaults to ModelScope `evalscope/deep-swe`
- DeepSWE runs through Pier's Docker environment in EvalScope
- Use `pier_agent_kwargs={'model_class': 'litellm'}` for OpenAI-compatible providers that do not support Responses API


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `deep_swe` |
| **Dataset ID** | [evalscope/deep-swe](https://modelscope.cn/datasets/evalscope/deep-swe/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `Coding`, `MultiTurn` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 113 |
| Prompt Length (Mean) | 2158.07 chars |
| Prompt Length (Min/Max) | 471 / 5385 chars |

## Sample Example

**Subset**: `test`

```json
{
  "input": [
    {
      "id": "f61040e0",
      "content": "Add a new `errorStack` constructor option to SuperJSON. Omitting it leaves existing Error behavior unchanged.\n\nThe option shape is `{ mode?, normalizeNewlines?, trimLeadingWhitespace?, maxStackLines?, stripInternalFrames?, redactPaths?, inclu ... [TRUNCATED 3577 chars] ... ): Processor | undefined`. `normalizeErrorStackOptions` returns `undefined` for any non-object input (`null`, `undefined`, strings).\n\nBefore writing, read through the existing error serialization logic and the `allowedErrorProps` mechanism.\n\n"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "ext_id": "kh701jywhzgddknqwzsq6npjv98226tq",
    "task_id": "superjson-error-stack-serialization",
    "display_title": "Add error stack serialization to SuperJSON",
    "display_description": "Add configurable serialization and restoration of error stacks, stack frames, causes, and sanitization in SuperJSON.",
    "repo": "flightcontrolhq/superjson",
    "repository_url": "https://github.com/flightcontrolhq/superjson.git",
    "original_title": "Error Stack Serialization Support",
    "category": "feature_request",
    "language": "typescript",
    "task_path": "~/.cache/evalscope/deep_swe/snapshots/evalscope/deep-swe/tasks/superjson-error-stack-serialization",
    "task_toml_path": "~/.cache/evalscope/deep_swe/snapshots/evalscope/deep-swe/tasks/superjson-error-stack-serialization/task.toml",
    "instruction": "Add a new `errorStack` constructor option to SuperJSON. Omitting it leaves existing Error behavior unchanged.\n\nThe option shape is `{ mode?, normalizeNewlines?, trimLeadingWhitespace?, maxStackLines?, stripInternalFrames?, redactPaths?, inclu ... [TRUNCATED 3577 chars] ... ): Processor | undefined`. `normalizeErrorStackOptions` returns `undefined` for any non-object input (`null`, `undefined`, strings).\n\nBefore writing, read through the existing error serialization logic and the `allowedErrorProps` mechanism.\n\n"
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
| `task_ids` | `list` | `[]` | Optional list of DeepSWE task ids to evaluate. |
| `languages` | `list` | `[]` | Optional task language filter from manifest metadata. |
| `categories` | `list` | `[]` | Optional task category filter from manifest metadata. |
| `sample_seed` | `int` | `` | Optional deterministic shuffle seed applied before limit. |
| `pier_agent_kwargs` | `dict` | `{}` | Extra kwargs passed to Pier AgentConfig.kwargs. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets deep_swe \
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
    datasets=['deep_swe'],
    dataset_args={
        'deep_swe': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


