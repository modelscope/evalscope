# Native AgentLoop Mode

Turn a regular benchmark (GSM8K, AIME, IFEval, SWE-bench, …) into a **multi-turn tool-use** task to evaluate the model's own tool-use and multi-step reasoning. EvalScope runs **generate → parse → call tool → observe → generate** on each iteration until the model submits a final answer or hits the step cap. The full trajectory is recorded as an `AgentTrace` for [replay in the UI](index.md#trace-visualization).

> To evaluate an off-the-shelf agent CLI such as Claude Code / Codex, see [External Agent Bridge Mode](bridge.md).

## When to use

- Check whether a model uses `python_exec` to verify its arithmetic on math problems.
- Let a ReAct-style model freely explore with the `bash` tool.
- Benchmark a model's multi-step interaction ability on SWE-bench and similar code-fix tasks.

## Quick start

Smallest runnable example — have `qwen-plus` verify GSM8K answers with `python_exec`.

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import AgentConfig

task_config = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key='<your-key>',
    eval_type='openai_api',
    datasets=['gsm8k'],
    limit=5,
    generation_config={'parallel_tool_calls': True},
    agent_config=AgentConfig(
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

## Three core components

Every AgentLoop is composed of three pieces, configured on `AgentConfig`:

**Strategy** — how the model talks to the loop:

| Name | Use when |
|------|----------|
| `function_calling` (default) | The model supports native function calling |
| `react` | Reasoning-first; let the model think before acting |
| `swe_bench_toolcall` | SWE-bench tool-call protocol (model supports function calling) |
| `swe_bench_backticks` | SWE-bench backticks text protocol (model lacks function calling) |

**Tools** — whitelist of capabilities exposed to the model:

| Name | Description |
|------|-------------|
| `bash` | Run shell commands |
| `python_exec` | Run Python code |
| `submit` | Submit the final answer (auto-injected by `function_calling` / `react`; do not configure manually) |

**Environment** — where tools actually execute:

| Name | Description |
|------|-------------|
| `local` | Host subprocess, no filesystem isolation — **dev / debug only** |
| `docker` | Container backed by [ms-enclave](https://github.com/modelscope/ms-enclave); recommended for production |

## Common configuration

Most-used `AgentConfig` fields:

| Field | Description | Recommended |
|-------|-------------|-------------|
| `strategy` | Which interaction protocol | `function_calling` (default) |
| `tools` | Tool whitelist | On demand, e.g. `['python_exec']` / `['bash']` |
| `environment` | Where tools run | `local` (dev) / `docker` (production) |
| `environment_extra` | Sandbox constructor kwargs | For `docker`: `{'image': '...', 'timeout': 60}`; see [Sandbox Environment](../sandbox.md) |
| `max_steps` | Max iterations per sample | Math/QA `5-10`, code fixes `100+` |
| `extra` | Strategy kwargs | Most commonly `{'system_prompt': '...'}` to steer the model |

```{tip}
- `local` has no isolation; use `docker` in production.
- `agent_config` also accepts a plain dict; `TaskConfig` converts it to `AgentConfig` automatically.
- With `docker` you reuse the [sandbox infrastructure](../sandbox.md); you may explicitly set `TaskConfig.sandbox = SandboxTaskConfig(enabled=True, engine='docker')`.
```

## MCP server tools

Beyond the built-in tools (`bash` / `python_exec` / ...), `AgentConfig` accepts arbitrary [MCP (Model Context Protocol)](https://modelcontextprotocol.io) servers via `mcp_servers`. Their advertised tools are listed at sample start, merged into the loop's tool set alongside the benchmark-native tools, and torn down when the sample finishes — **no benchmark-side change required**.

This is how to plug in `fetch`, web search, GitHub, filesystem and the rest of the MCP ecosystem without writing a custom EvalScope tool.

### Install

```bash
pip install evalscope[mcp]
# Plus whichever MCP servers you want to use, e.g.:
pip install mcp-server-fetch
```

The `mcp` Python SDK is imported lazily, so configurations with empty `mcp_servers` (the default) need no extra install.

### Quick start

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
            # Stdio: spawn a local Python MCP server.
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
            # Stdio: use uvx so end users don't have to pip-install first.
            MCPServerConfigStdio(
                command='uvx',
                args=['mcp-server-brave-search'],
                env={'BRAVE_API_KEY': 'sk-...'},
                name='brave_search',
            ),
            # HTTP: connect to a remote MCP endpoint.
            MCPServerConfigHTTP(
                url='https://my-mcp.example.com',
                headers={'Authorization': 'Bearer ...'},
                name='internal',
            ),
        ],
    ),
    dataset_args={'gaia': {'subset_list': ['2023_level1']}},
    limit=5,
)
run_task(task_config)
```

Inside the agent loop the model now sees `bash`, `submit`, **and** every tool the MCP servers advertised (e.g. `fetch`, `brave_web_search`). It picks tools by name as usual.

### Configuration

| Field | Type | Description |
|-------|------|-------------|
| `command` (stdio) | str | Executable to spawn (`sys.executable` / `uvx` / `npx` / absolute path). |
| `args` (stdio) | list[str] | Arguments passed to `command`. |
| `env` (stdio) | dict[str, str] | Extra environment variables for the child process. |
| `cwd` (stdio) | str | Working directory for the child. |
| `url` (http) | str | Streamable HTTP endpoint of the MCP server. |
| `headers` (http) | dict[str, str] | HTTP headers (typically auth). |
| `timeout` (http) | float | HTTP read timeout in seconds (default `30.0`). |
| `name` | str | Display name used in logs and the trace UI. Defaults to `command` / `url`. |
| `tools` | `'all'` or list[str] | Whitelist; `'all'` (default) exposes every tool the server advertises. |

```{tip}
- `python -m mcp_server_<x>` after `pip install` is the most deterministic for CI/tests — no per-run package fetch.
- `uvx mcp-server-<x>` is convenient for ad-hoc runs; first call downloads the package.
- For benchmarks that already drive their own loop (`AgentLoopAdapter` subclasses like GAIA / SWE-bench-agentic), MCP servers from the global `agent_config` are still picked up via `agent_config.mcp_servers` and merged into the loop's tool set.
```

### Trace / debugging

`MCPServer[<name>]: initialised` and `MCPServer[<name>]: closed` log lines bracket the per-sample server lifecycle. Tool calls appear in `agent_trace.events` like any other `tool_call` / `tool_result` pair — the tool `name` is the MCP-advertised name.

If a MCP tool call fails, the observation comes back as `[error] <body>` so the model can recover and try a different approach without crashing the loop.

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
                'action_protocol': 'toolcall',  # or 'backticks' if the model lacks function calling
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

Common `extra_params` keys: `action_protocol`, `max_steps`, `command_timeout`, `working_dir`, `build_docker_images`, `pull_remote_images_if_available`, `force_arch`. Full reference in the [SWE-bench dataset docs](../../third_party/swe_bench.md).

```{important}
Prerequisites: `pip install evalscope[swe_bench]`, Docker installed locally. A globally configured `agent_config` is **ignored** for these benchmarks.
```

## FAQ

**The model never calls any tool**

1. Check tool-name spelling in `tools`.
2. Confirm the model supports function calling; if not, switch to a text protocol like `swe_bench_backticks`.
3. Inspect `agent_trace.events` for `nudge` (system already reminded the model) or `error` (parse / execution failure).
4. Use `kwargs={'system_prompt': '...'}` to explicitly require a specific tool.

**How big should `max_steps` be?**

- Math / simple QA: `3`–`10` is usually enough.
- Code fixes / SWE-bench: keep the default `250`; lower values produce a lot of `max_steps_exceeded`.
