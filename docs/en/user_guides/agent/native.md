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
