# WideSearch

## Introduction

[WideSearch](https://github.com/ByteDance-Seed/WideSearch) evaluates whether a search agent can collect a broad,
complete set of facts from the web and return them in a required Markdown table. The ModelScope dataset
[`bytedance-community/WideSearch`](https://modelscope.cn/datasets/bytedance-community/WideSearch) contains 200 tasks in
the `full` split, with 100 English and 100 Chinese tasks. Each task includes a gold CSV and a per-column evaluation
configuration.

EvalScope implements the official single-agent setting. It uses the official language-specific system prompt,
`function_calling`, bash by default, and the official table alignment and hybrid rule/LLM scoring pipeline. The
official multi-agent `create_sub_agents` baseline is not included.

## Installation

```bash
pip install 'evalscope[wide_search]'
```

Install optional runtime integrations as needed:

```bash
pip install 'evalscope[wide_search,mcp]'      # Fetch MCP and other MCP servers
pip install 'evalscope[wide_search,sandbox]'  # Docker sandbox
```

## Default Local Evaluation

The default environment is a temporary local working directory with host network access. It is cleaned after each
sample, but it is not a security sandbox: absolute paths can still access host files. Use it only with trusted models.

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    api_url='OPENAI_COMPATIBLE_URL',
    api_key=os.getenv('MODEL_API_KEY'),
    eval_type='openai_api',
    datasets=['wide_search'],
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_JUDGE_MODEL',
        'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
        'api_key': os.getenv('JUDGE_API_KEY'),
        'generation_config': {'temperature': 0.0},
    },
    eval_batch_size=1,
    limit=1,  # Remove for the full 200-task evaluation
))
```

The benchmark defaults to 50 AgentLoop steps and a 120-second bash timeout. Override them with
`NativeAgentConfig(max_steps=..., command_timeout=...)`.

## Docker Environment

Enable the unified EvalScope sandbox configuration to run bash inside Docker. The default image is
`python:3.11-slim` with network access.

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.config import SandboxTaskConfig

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['wide_search'],
    agent_config=NativeAgentConfig(max_steps=50, command_timeout=120),
    sandbox=SandboxTaskConfig(
        enabled=True,
        default_config={
            'image': 'python:3.11-slim',
            'network_enabled': True,
        },
    ),
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
    limit=1,
))
```

Docker mode requires a running Docker daemon and `evalscope[sandbox]`.

## Fetch MCP and Paper-style Repeats

MCP is optional. The following configuration keeps bash, adds the official Fetch MCP server, and uses four trials per
task to produce the paper's `Avg@4`, `Pass@4`, and `Max@4` report shape.

```python
import sys

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['wide_search'],
    repeats=4,
    agent_config=NativeAgentConfig(
        max_steps=50,
        command_timeout=120,
        mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            )
        ],
    ),
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```

## Scoring and Reports

For each task, the scorer parses the Markdown table, normalizes columns, uses the judge to align semantically equivalent
column names and primary-key entities, joins prediction and gold rows, and applies the configured preprocessors and
metrics. Supported official operations include `norm_str`, `extract_number`, `norm_date`, `exact_match`, `number_near`,
`date_near`, `url_match`, and `llm_judge`.

Each trial produces seven metrics:

- `success_rate`
- `row_precision`, `row_recall`, `row_f1`
- `item_precision`, `item_recall`, `item_f1`

A single full run derives three report scopes without repeating inference: `all`, `en`, and `zh`. Success rate reports
`Avg@N` and `Pass@N`; row/item metrics report `Avg@N` and `Max@N`. Use `repeats=4` for the paper-style report shape.

The judge is also used for schema/entity alignment, so rule-only evaluation is intentionally unsupported. The paper
recommends GPT-4.1-2025-04-14 for comparable judging, but EvalScope does not hard-code a judge model.

## Local Dataset

To avoid a remote download, set `dataset_args={'wide_search': {'local_path': '/path/to/WideSearch'}}`. The directory
must contain:

```text
WideSearch/
├── widesearch.jsonl
└── widesearch_gold/
    ├── ws_en_001.csv
    └── ...
```

## Resources

- [Paper](https://arxiv.org/abs/2508.07999)
- [Official repository](https://github.com/ByteDance-Seed/WideSearch)
- [ModelScope dataset](https://modelscope.cn/datasets/bytedance-community/WideSearch)
