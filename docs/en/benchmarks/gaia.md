# GAIA


## Overview

GAIA (General AI Assistants) is a benchmark of 450+ questions targeting next-generation LLMs with tool use, web browsing and multi-step reasoning. Each question has an unambiguous short answer and is bucketed into one of three difficulty levels.

## Task Description

- **Task Type**: Tool-use Agent (multi-turn)
- **Input**: Natural-language question, optionally with one referenced attachment file (PDF / xlsx / image / audio / ...)
- **Output**: A short final answer (number / short phrase / comma-separated list)
- **Splits**: ``validation`` (answers public) and ``test`` (answers private — evaluation skipped)

## Key Features

- ReAct agent loop with a single ``bash`` tool inside a Docker sandbox (default image ``python:3.11``, includes ``curl`` / ``wget`` / ``git``).
- Attachment files are mounted read-only at ``/shared_files`` inside the sandbox.
- Rule-based scorer ported verbatim from the official GAIA leaderboard (no LLM judge).
- Dataset downloaded from ModelScope (``gaia-benchmark/GAIA``) by default; set ``dataset_hub='huggingface'`` to load from Hugging Face instead.

## Evaluation Notes

- Requires Docker daemon running locally (or a remote sandbox engine via the ms_enclave configuration).
- ``extra_params.max_steps`` caps the agent loop length (default 50).
- ``extra_params.command_timeout`` sets per-``bash`` command timeout (default 180s, mirrors inspect_ai).
- Network is enabled by default — many questions require browsing/searching.
- Use ``subset_list`` to restrict to specific difficulty levels, e.g. ``['2023_level1']``, ``['2023_level1', '2023_level2']`` or ``['2023_all']`` (default).
- [Usage Documentation](https://evalscope.readthedocs.io/en/latest/third_party/gaia.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gaia` |
| **Dataset ID** | [gaia-benchmark/GAIA](https://modelscope.cn/datasets/gaia-benchmark/GAIA/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `MultiTurn`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 165 |
| Prompt Length (Mean) | 861.35 chars |
| Prompt Length (Min/Max) | 596 / 2582 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `2023_level1` | 53 | 906.53 | 604 | 2582 |
| `2023_level2` | 86 | 816.66 | 596 | 1275 |
| `2023_level3` | 26 | 917.08 | 621 | 1497 |

## Sample Example

**Subset**: `2023_level1`

```json
{
  "input": [
    {
      "id": "93e8257c",
      "content": "Please answer the question below. You should:\n\n- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.\n- If the answer is a number, return only ... [TRUNCATED 431 chars] ... Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."
    }
  ],
  "target": "17",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "bash",
      "description": "Execute a bash command inside the sandbox environment. Returns the combined stdout / stderr output of the command.",
      "parameters": {
        "properties": {
          "command": {
            "type": "string",
            "description": "The bash command to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "command"
        ]
      }
    }
  ],
  "metadata": {
    "task_id": "e1fc63a2-da7a-432f-be78-7c4a95598703",
    "level": "1",
    "file_name": "",
    "file_path": "",
    "Annotator Metadata": {
      "Steps": "1. Googled Eliud Kipchoge marathon pace to find 4min 37sec/mile\n2. Converted into fractions of hours.\n3. Found moon periapsis in miles (225,623 miles).\n4. Multiplied the two to find the number of hours and rounded to the nearest 100 hours.",
      "Number of steps": "4",
      "How long did this take?": "20 Minutes",
      "Tools": "1. A web browser.\n2. A search engine.\n3. A calculator.",
      "Number of tools": "3"
    }
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
| `max_steps` | `int` | `50` | Maximum number of agent steps per sample. |
| `command_timeout` | `float` | `180.0` | Default per-bash-command timeout in seconds. |
| `docker_image` | `str` | `python:3.11` | Docker image used as the per-sample sandbox. |
| `network_enabled` | `bool` | `True` | Allow the sandbox to access the network (GAIA browsing questions need this). |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gaia \
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
    datasets=['gaia'],
    dataset_args={
        'gaia': {
            # subset_list: ['2023_level1', '2023_level2', '2023_level3']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


