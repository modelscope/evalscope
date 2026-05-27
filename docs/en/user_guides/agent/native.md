# Native AgentLoop Mode

Turn a regular benchmark (GSM8K, AIME, IFEval, SWE-bench, …) into a **multi-turn tool-use** task to evaluate the model's own tool-use and multi-step reasoning. EvalScope runs **generate → parse → call tool → observe → generate** on each iteration until the model submits a final answer or hits the step cap. The full trajectory is recorded as an `AgentTrace` for [replay in the UI](index.md#trace-visualization).

> To evaluate an off-the-shelf agent CLI such as Claude Code / Codex, see [External Agent Bridge Mode](bridge.md).

## When to use

- **Evaluate a model's multi-step reasoning and tool-use ability**
  Example: does qwen-plus proactively call `python_exec` to verify arithmetic on GSM8K? Can it reason step-by-step and self-check on AIME?

- **Let a model autonomously explore code-fix tasks**
  Example: on SWE-bench, give the model a `bash` tool to read code, run tests, and submit patches — no human in the loop.

- **Plug in MCP ecosystem tools to expand model capabilities**
  Example: attach `fetch`, Brave Search, or other MCP servers so the model can search the web when running benchmarks like GAIA.

## Quick start

Smallest runnable example — have `qwen-plus` verify GSM8K answers with `python_exec`.

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_config = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key='<your-key>',
    eval_type='openai_api',
    datasets=['gsm8k'],
    limit=5,
    generation_config={'parallel_tool_calls': True},
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        tools=['python_exec'],
        environment='docker',
        environment_extra={'image': 'python:3.11-slim'},
        max_steps=5,
        kwargs={'system_prompt': 'Use python_exec to verify your calculations.'},
    ),
)
run_task(task_config)
```

EvalScope auto-injects the `python_exec` tool definition into the model's context, executes the generated Python in a Docker container, feeds stdout/stderr back as an observation, and loops until the model uses `submit` to deliver its answer.

## Common configuration

Most-used `NativeAgentConfig` fields:

| Field | Description | Recommended |
|-------|-------------|-------------|
| `strategy` | Interaction protocol | `function_calling` (default); use `react` or `swe_bench_backticks` if the model lacks function-calling |
| `tools` | Tool whitelist | On demand, e.g. `['python_exec']` / `['bash']` |
| `environment` | Where tools run | `local` (dev) / `docker` (production) |
| `environment_extra` | Sandbox constructor kwargs | For `docker`: `{'image': '...', 'timeout': 60}`; see [Sandbox Environment](../sandbox.md) |
| `max_steps` | Max iterations per sample | Math/QA `5-10`, code fixes `100+` |
| `kwargs` | Strategy kwargs | Most commonly `{'system_prompt': '...'}` to steer the model |

Available `strategy` values:

| Name | Use when |
|------|----------|
| `function_calling` (default) | The model supports native function calling |
| `react` | Reasoning-first; let the model think before acting |
| `swe_bench_toolcall` | SWE-bench tool-call protocol |
| `swe_bench_backticks` | SWE-bench backticks text protocol (model lacks function calling) |

Available tools (`tools`):

| Name | Description |
|------|-------------|
| `bash` | Run shell commands |
| `python_exec` | Run Python code |
| `submit` | Submit the final answer (auto-injected; do not configure manually) |

```{tip}
- `local` has no filesystem isolation; use `docker` in production.
- `agent_config` also accepts a plain dict; `TaskConfig` converts it to `NativeAgentConfig` automatically.
```

## MCP server tools

Beyond the built-in tools, `NativeAgentConfig` accepts arbitrary [MCP (Model Context Protocol)](https://modelcontextprotocol.io) servers via `mcp_servers` — plug in `fetch`, web search, GitHub, and the rest of the MCP ecosystem without writing a custom EvalScope tool.

### Install

```bash
pip install evalscope[mcp]
```

`evalscope[mcp]` bundles the official `mcp` Python SDK plus `mcp-server-fetch` (universal HTTP fetching, no API key). Other MCP servers can be installed on demand or run via `uvx` / `npx`.

### Quick start

Attach HTTP fetching and a remote MCP tool to a GAIA evaluation:

```python
import sys
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio, MCPServerConfigHTTP

task_config = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    eval_type='openai_api',
    datasets=['gaia'],
    agent_config=NativeAgentConfig(
        mcp_servers=[
            # Spawn a local MCP server process
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
            # Connect to a remote MCP endpoint
            MCPServerConfigHTTP(
                url='https://mcp.api-inference.modelscope.net/<server-id>/mcp',
                headers={'Authorization': 'Bearer <TOKEN>'},
                name='modelscope_tool',
            ),
        ],
    ),
    dataset_args={'gaia': {'subset_list': ['2023_level1']}},
    limit=5,
)
run_task(task_config)
```

Once configured, the model sees both built-in tools and every tool advertised by the MCP servers, picking them by name. If an MCP tool call fails, the error is fed back to the model so it can retry — the evaluation loop keeps running.

### Connection methods

EvalScope supports three MCP transports:

| Config class | When to use | Key parameters |
|--------------|------------|----------------|
| `MCPServerConfigStdio` | Spawn a local MCP server process | `command`, `args`, `env` |
| `MCPServerConfigHTTP` | Connect to a remote Streamable HTTP endpoint (recommended) | `url`, `headers` |
| `MCPServerConfigSSE` | Connect to a legacy SSE endpoint | `url`, `headers` |

All config classes share `name` (display name in logs) and `tools` (tool whitelist, defaults to `'all'`). For the full parameter list, see the [source](https://github.com/modelscope/evalscope/tree/main/evalscope/api/agent/mcp).

```{tip}
- For CI/production, prefer `python -m mcp_server_<x>` after `pip install` — no per-run package fetch.
- `uvx mcp-server-<x>` is convenient for ad-hoc runs; first call downloads the package.
- When a remote service offers both `/mcp` and `/sse`, prefer `/mcp` (Streamable HTTP).
```

## SWE-bench agentic benchmarks

The `swe_bench_*_agentic` family (`swe_bench_verified_agentic`, `swe_bench_verified_mini_agentic`, `swe_bench_lite_agentic`) ships its own AgentLoop and **ignores** the global `agent_config`. All loop parameters go through `dataset_args.extra_params`.

```python
task_config = TaskConfig(
    model='qwen-plus',
    api_url='...',
    api_key='...',
    eval_type='openai_api',
    datasets=['swe_bench_verified_mini_agentic'],
    dataset_args={
        'swe_bench_verified_mini_agentic': {
            'extra_params': {
                'action_protocol': 'toolcall',  # or 'backticks'
                'max_steps': 250,
                'command_timeout': 60.0,
            },
        },
    },
    limit=3,
    generation_config={'max_tokens': 4096, 'temperature': 0.0},
)
run_task(task_config)
```

Common `extra_params` keys: `action_protocol`, `max_steps`, `command_timeout`, `working_dir`. Full reference in the [SWE-bench dataset docs](../../third_party/swe_bench.md).

```{important}
Prerequisites: `pip install evalscope[swe_bench]`, Docker installed locally.
```

## FAQ

**The model never calls any tool**

1. Check tool-name spelling in `tools`.
2. Confirm the model supports function calling; if not, switch to a text protocol like `react` or `swe_bench_backticks`.
3. Use `kwargs={'system_prompt': '...'}` to explicitly steer the model toward tool use.

**How big should `max_steps` be?**

- Math / simple QA: `3`–`10` is usually enough.
- Code fixes / SWE-bench: keep the default `250`; lower values produce a lot of `max_steps_exceeded`.
