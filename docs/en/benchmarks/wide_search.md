# WideSearch


## Overview

WideSearch evaluates search agents on broad information-seeking tasks: collecting a large, complete set of atomic facts
from the live web and organizing them into one strictly structured Markdown table. The benchmark contains 200 manually
curated tasks, evenly split between English and Chinese.

## Task and Runtime

- **Input**: A natural-language collection request with an explicit Markdown table schema
- **Output**: One complete Markdown table
- **Agent**: EvalScope's benchmark-owned AgentLoop with the official single-agent prompt and function-calling strategy
- **Tools**: Bash by default; optional MCP servers are merged from ``NativeAgentConfig.mcp_servers``
- **Environment**: Per-sample temporary local directory by default, optionally a Docker sandbox

The local environment uses the host network and is not a security sandbox: absolute paths can still access host files.
Use it only with trusted models. Set ``environment_type='docker'`` for isolation, or attach search and page-fetching MCP
servers for a more representative search-agent setup. The official multi-agent ``create_sub_agents`` baseline is not
implemented by this adapter.

## Evaluation

The scorer preserves the official table alignment and hybrid cell-level evaluation pipeline. It parses the Markdown
table, aligns column names and primary-key entities with an LLM judge, then applies per-column exact, numerical, date,
URL, or semantic matching. Rule-only judging is not supported; provide ``judge_model_args`` with
``judge_strategy='auto'`` or ``'llm'``. GPT-4.1-2025-04-14 is recommended for comparison with the paper.

Each trial reports table success rate and row/item precision, recall, and F1. Repeated trials are aggregated into the
paper's Avg@N, Pass@N, and Max@N metrics for ``all``, ``en``, and ``zh`` without evaluating samples more than once.
Use ``repeats=4`` to reproduce the paper's reporting shape.

## Configuration

- ``max_steps``: 50 by default
- ``environment_type``: ``local`` (default) or ``docker``
- ``command_timeout``: 120 seconds for bash commands
- ``docker_image``: ``python:3.11-slim``
- ``network_enabled``: enabled by default for Docker

Resources: [Paper](https://arxiv.org/abs/2508.07999) |
[GitHub](https://github.com/ByteDance-Seed/WideSearch) |
[Dataset](https://modelscope.cn/datasets/bytedance-community/WideSearch)

## Usage

Default local bash evaluation:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```

Choose the local or Docker environment through benchmark parameters:

```python
TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    dataset_args={'wide_search': {'extra_params': {'environment_type': 'docker'}}},
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
)
```

Reproduce the paper's four-trial report shape with bash and the Fetch MCP server:

```python
import sys

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    repeats=4,
    agent_config=NativeAgentConfig(mcp_servers=[MCPServerConfigStdio(
        command=sys.executable,
        args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
        name='fetch',
    )]),
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `wide_search` |
| **Dataset ID** | [bytedance-community/WideSearch](https://modelscope.cn/datasets/bytedance-community/WideSearch/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2508.07999) |
| **Tags** | `Agent`, `MultiTurn`, `Retrieval` |
| **Metrics** | `success_rate`, `row_precision`, `row_recall`, `row_f1`, `item_precision`, `item_recall`, `item_f1` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `full` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 200 |
| Prompt Length (Mean) | 631.01 chars |
| Prompt Length (Min/Max) | 182 / 1807 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "cf605347",
      "content": "My son is about to start his university applications in 2025 for postgraduates but he’s still uncertain about both his major and which universities to apply to. Could you help me find the top five universities in each of the five broad subjec ... [TRUNCATED 691 chars] ... names in English. \nUse only Arabic numerals in the ranking, for example: 1.\nDon't ask me any questions, just output the results according to the columns without omitting cells arbitrarily. The output format is \n```markdown\n{data_content}\n```."
    }
  ],
  "target": "﻿Subject,University,Country,QS World University Rankings by Subject 2025,QS World University Rankings 2025,Times Higher Education  World University Rankings 2025,Home Page,Application Deadline,Application Fee\nArts & Humanities,Harvard Univers ... [TRUNCATED 2838 chars] ... nuary 5,$90\nSocial Sciences & Management,Massachusetts Institute of Technology,United States,4,1,2,https://www.mit.edu/,January 6 ,$75\nSocial Sciences & Management,University of Cambridge,United Kingdom,5,5,5,https://www.cam.ac.uk/,Oct 15,£60",
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
    "instance_id": "ws_en_001",
    "language": "en",
    "evaluation": {
      "unique_columns": [
        "subject",
        "university"
      ],
      "required": [
        "subject",
        "university",
        "qsworlduniversityrankingsbysubject2025",
        "qsworlduniversityrankings2025",
        "timeshighereducationworlduniversityrankings2025",
        "homepage",
        "applicationdeadline",
        "applicationfee"
      ],
      "eval_pipeline": {
        "applicationdeadline": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nThe month and day must be correct"
        },
        "applicationfee": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nIf there are multiple fees in the reference answer, all must be included."
        },
        "homepage": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "url_match"
          ]
        },
        "subject": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "university": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankingsbysubject2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "timeshighereducationworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        }
      }
    }
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `int` | `50` | Maximum number of function-calling AgentLoop steps per sample. |
| `environment_type` | `str` | `local` | Environment used by the built-in bash tool. Choices: ['local', 'docker'] |
| `command_timeout` | `float` | `120.0` | Default timeout for bash commands in local or Docker environments. |
| `docker_image` | `str` | `python:3.11-slim` | Docker image used when environment_type is docker. |
| `network_enabled` | `bool` | `True` | Allow network access in the Docker environment. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets wide_search \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":50}' \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['wide_search'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=50,
    ),
    dataset_args={
        'wide_search': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
