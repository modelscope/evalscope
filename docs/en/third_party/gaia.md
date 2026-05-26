# GAIA

## Introduction

[GAIA](https://arxiv.org/abs/2311.12983) (General AI Assistants) is a benchmark of 450+ questions designed to evaluate next-generation LLMs equipped with tool use, web browsing and multi-step reasoning. Each question has a single unambiguous short answer (a number, a short phrase, or a comma-separated list).

Questions are split into three difficulty levels:

| Level | Description |
|-------|-------------|
| `2023_level1` | Should be solvable by capable LLMs with basic tool use |
| `2023_level2` | Requires more autonomous planning and richer tool use |
| `2023_level3` | Indicates a strong jump in agent capability — multi-modal sources, long chains of reasoning |

Each level provides a public `validation` split (with answers) and a private `test` split (answers withheld for the [official leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)). EvalScope currently supports **`validation` only**.

GAIA's evalscope integration drives a multi-turn ReAct agent loop inside a per-sample Docker container, with a single `bash` tool. The official rule-based scorer (number / list / string normalization) is ported verbatim from the GAIA leaderboard.

## Install Dependencies

GAIA evaluation runs the agent inside Docker:

1. **Install Docker**: see the [Docker Installation Guide](https://docs.docker.com/engine/install/). Make sure the daemon is running.
2. **Install evalscope**:
   ```bash
   pip install evalscope
   ```

The first run will pull the `python:3.11` image (~1.1GB) and download the GAIA dataset snapshot from ModelScope (~110MB) — these are cached for subsequent runs.

```{note}
The benchmark uses the [`gaia-benchmark/GAIA`](https://www.modelscope.cn/datasets/gaia-benchmark/GAIA) ModelScope mirror by default — no HuggingFace gating, no token required. Set `dataset_hub='huggingface'` if you prefer the original repo (you'll need to accept the dataset terms first).
```

## Datasets

A single benchmark `gaia` covers all three difficulty levels via `subset_list`:

| Configuration | Loads |
|---------------|-------|
| `subset_list=['2023_level1']` | Level 1 only |
| `subset_list=['2023_level1', '2023_level2']` | Levels 1 + 2 |
| `subset_list=['2023_level1', '2023_level2', '2023_level3']` (default) | All three levels |

About 1/3 of GAIA questions reference an attachment file (PDF / xlsx / image / audio / ...). EvalScope mounts the dataset's `2023/validation/` directory **read-only** into the sandbox at `/shared_files`, so the agent can access referenced files via the path hint embedded in the prompt.

## Run Example

The example below mirrors `test_gaia` in [tests/benchmark/test_agent.py](https://github.com/modelscope/evalscope/blob/main/tests/benchmark/test_agent.py):

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['gaia'],
    dataset_args={
        'gaia': {
            'subset_list': ['2023_level1'],            # Pick which level(s) to run
            'extra_params': {
                'max_steps': 50,                       # Maximum agent loop steps per sample
                'command_timeout': 180.0,              # Per-bash-command timeout in seconds
                'docker_image': 'python:3.11',         # Sandbox image; full image bundles curl/wget/git
                'network_enabled': True,               # Required: most questions need network access
            }
        }
    },
    eval_batch_size=5,  # Number of parallel sandboxes
    limit=5,            # Limit samples for quick testing; remove for the full validation set
    generation_config={
        'temperature': 0.7,
        'parallel_tool_calls': True,
        'stream': True,
    }
)
run_task(task_cfg=task_cfg)
```

Example final result:

```text
+-----------+---------+----------+-------------+-------+---------+---------+
| Model     | Dataset | Metric   | Subset      |   Num |   Score | Cat.0   |
+===========+=========+==========+=============+=======+=========+=========+
| qwen3-max | gaia    | mean_acc | 2023_level1 |     5 |     0.2 | default |
+-----------+---------+----------+-------------+-------+---------+---------+
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | int | `50` | Maximum agent loop steps per sample. Mirrors inspect_ai's `message_limit=100` (~50 ReAct turns). |
| `command_timeout` | float | `180.0` | Per-bash-command timeout in seconds. |
| `docker_image` | str | `python:3.11` | Sandbox image. The full `python:3.11` ships `curl` / `wget` / `git`; use `python:3.11-slim` (~130MB) only if you bake your own tools. |
| `network_enabled` | bool | `True` | Allow the sandbox to access the network. Most GAIA questions need it. |

```{note}
GAIA spawns one Docker container per sample. Each container is destroyed once the sample completes. Tune `eval_batch_size` according to your machine resources — each in-flight sample holds a container open.
```

## Scoring

The scorer is a verbatim port of the [official GAIA leaderboard scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py) (Apache 2.0):

- **Numeric ground truth**: strip `$` / `%` / `,`, parse as float, compare for exact equality.
- **List ground truth** (contains `,` or `;`): split on those delimiters, compare element-wise (number-aware).
- **String ground truth**: strip whitespace, lowercase, remove punctuation, compare for equality.

No LLM judge is involved. The agent must produce an answer that **exactly matches** the canonical normalization above — typically by calling the auto-injected `submit(answer=...)` tool.

## Adding web access via MCP

GAIA's bash-only sandbox can `curl` / `wget` raw HTML, but cannot run a JS-rendered browser or hit gated search APIs. The simplest way to give the agent real browsing power is to plug an MCP server (e.g. `mcp-server-fetch` for HTTP, `mcp-server-brave-search` for keyword search) — the host-side MCP servers run **outside** the per-sample Docker, no sandbox image change needed.

```bash
pip install evalscope[mcp]
```

`evalscope[mcp]` ships with `mcp-server-fetch` already bundled, so no further install is needed for the example below.

```python
import sys
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    eval_type='openai_api',
    datasets=['gaia'],
    dataset_args={'gaia': {'subset_list': ['2023_level1']}},
    agent_config=NativeAgentConfig(
        mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                # ``--ignore-robots-txt`` avoids stalling when the upstream
                # robots.txt request is intermittently blocked.
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
        ],
    ),
    limit=5,
)
run_task(task_cfg)
```

The agent now sees `bash`, `submit` and `fetch` as tools and can call `fetch(url=...)` to read web pages directly — no docker-side library install, no `curl` plumbing.

See [Native Agent Loop → MCP server tools](../user_guides/agent/native.md#mcp-server-tools) for the full configuration reference (stdio / HTTP transports, tool whitelist, env vars, etc.).

## Known Limitations

- **No `web_search` / `web_browser` tool yet**: GAIA Phase-1 ships only the `bash` tool. The agent can `curl` / `wget` and parse HTML with `python3`, but JavaScript-rendered pages and gated search APIs cannot be reached. Browsing-heavy questions will score significantly lower than implementations with a real browser tool.
- **No support for `test` split**: only `validation` (which has public answers) is supported. To submit to the official leaderboard you'll need to capture model predictions and format them yourself.
- **Long bash outputs may exceed model input length**: a `curl` of a large HTML page accumulates fast in the agent's message history and can hit the model's max input length on harder samples. Set `ignore_errors=True` on `TaskConfig` (or in the test) so the rest of the run continues when that happens.
