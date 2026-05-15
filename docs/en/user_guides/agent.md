# Agent Evaluation Mode

EvalScope provides an **Agent Evaluation Mode** that lets the model complete evaluation tasks inside a controlled multi-turn tool-calling loop (AgentLoop). It is designed for evaluating end-to-end agentic capabilities such as tool use, multi-step reasoning, and code repair, and records a full interaction trajectory (AgentTrace) for every sample so that runs can be replayed and visualized.

## Overview

There are two ways to enable Agent mode:

- **Global switch**: set `agent_config = AgentConfig(...)` on `TaskConfig`. All benchmarks based on `DefaultDataAdapter` (GSM8K, AIME, IFEval, etc.) automatically switch to AgentLoop for inference.
- **Benchmark-built-in**: a few benchmarks (e.g. `swe_bench_*_agentic`) are driven directly by `AgentLoopAdapter` subclasses; their loop parameters live under `dataset_args.extra_params` and **do not read** the global `agent_config`.

In each iteration AgentLoop performs **generate → parse → tool call → observe → generate again**, until the model explicitly submits a final answer or `max_steps` is reached. The whole flow is composed of three pluggable parts:

| Component | Meaning | Built-in implementations |
|-----------|---------|--------------------------|
| **Strategy** | Controls system prompt, message formatting, output parsing and termination | `function_calling`, `react`, `swe_bench_toolcall`, `swe_bench_backticks` |
| **Tool** | Capability the model can invoke; exposed as a tool protocol or text block by the strategy | `bash`, `python_exec`, `submit` (auto-injected) |
| **Environment** | The isolated sandbox where tools actually run; provides `exec` / `close` interfaces | `local` (subprocess), `docker` (container isolation) |

Key events of every step (model generation, tool call, tool result, environment command, error, nudge, submit) are recorded into `AgentTrace.events` in a structured form, and persisted alongside the evaluation result in the `agent_trace` field of `reviews/<model_id>/<dataset>.jsonl` so that the HTML report and the Web dashboard can replay them.

## AgentConfig Reference

`AgentConfig` is defined in `evalscope.api.agent` and is the value type of `TaskConfig.agent_config`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `str` | `function_calling` | Registered strategy name; controls how the model interacts with the loop |
| `tools` | `list[str]` | `[]` | Whitelist of tools available to the model. Empty list means pure multi-turn chat with no tools |
| `environment` | `str \| None` | `None` | Registered environment name. `None` means no sandbox (only suitable for tool-less or side-effect-free tools) |
| `max_steps` | `int` | `10` | Hard upper bound of loop iterations. Exceeding it logs `max_steps_exceeded` and exits |
| `extra` | `dict` | `{}` | Strategy constructor kwargs (e.g. `system_prompt`) |
| `environment_extra` | `dict` | `{}` | Environment constructor kwargs (e.g. `image`, `timeout`) |

### 2.1 Available strategies

| Name | Description |
|------|-------------|
| `function_calling` | Default strategy that uses the model's native function-calling API. Auto-injects a `submit` tool for explicit termination |
| `react` | ReAct-style strategy that injects a reasoning-first system prompt; also auto-injects the `submit` tool |
| `swe_bench_toolcall` | SWE-bench tool-calling protocol; only allows the `bash` tool, terminates via the `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel string |
| `swe_bench_backticks` | SWE-bench backticks text protocol, suitable for models that do not support function-calling |

### 2.2 Available tools

| Name | Description | Requires environment |
|------|-------------|----------------------|
| `bash` | Executes shell commands and returns stdout / stderr | Yes |
| `python_exec` | Executes Python source code and returns stdout / stderr | Yes |
| `submit` | Explicitly submits a final answer; auto-injected by `function_calling` and `react`, **no manual configuration needed** | No |

### 2.3 Available environments

| Name | Description |
|------|-------------|
| `local` | `LocalAgentEnvironment`, runs commands as subprocesses; no filesystem isolation, **only recommended for development and unit tests** |
| `docker` | `EnclaveAgentEnvironment`, launches a Docker container via [ms-enclave](https://github.com/modelscope/ms-enclave) for full isolation |

### 2.4 environment_extra fields

- **local environment**:
  - `working_dir` (`str`): working directory
  - `env_vars` (`dict[str, str]`): environment variables to inject
- **docker environment**:
  - `image` (`str`): Docker image name (e.g. `python:3.11-slim`)
  - `timeout` (`float`): per-command timeout (seconds)
  - `environment` (`dict[str, str]`): container environment variables

```{tip}
When `environment='docker'` is enabled, EvalScope reuses the ms-enclave infrastructure described in [Sandbox Environment](sandbox.md). Make sure the sandbox dependencies and Docker are installed first, and we recommend explicitly turning on the sandbox via `TaskConfig.sandbox = SandboxTaskConfig(enabled=True, engine='docker')` (see the sandbox docs for details).
```

### 2.5 extra fields

- `system_prompt` (`str`): override the strategy's default system prompt, e.g. force the model to call `python_exec` for calculations.

## Running Generic Benchmarks in Agent Mode

### 3.1 Case A: GSM8K + function_calling + python_exec + Docker

Force the model to call `python_exec` to verify its calculations while solving GSM8K problems, with the tool running inside a Docker container.

```python
from dotenv import dotenv_values
from evalscope import TaskConfig, run_task
from evalscope.api.agent import AgentConfig

env = dotenv_values('.env')

task_config = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['gsm8k'],
    dataset_args={'gsm8k': {'few_shot_num': 0}},
    eval_batch_size=5,
    limit=5,
    generation_config={
        'max_tokens': 2048,
        'temperature': 0.7,
        'parallel_tool_calls': True,
    },
    agent_config=AgentConfig(
        strategy='function_calling',
        tools=['python_exec'],
        environment='docker',
        environment_extra={'image': 'python:3.11-slim', 'timeout': 60},
        max_steps=5,
        extra={
            'system_prompt': (
                'You are a math solver. '
                'Use the python_exec tool to verify your calculations.'
            ),
        },
    ),
)

run_task(task_config)
```

After a successful run, every record in `reviews/qwen-plus/gsm8k_*.jsonl` will have an `agent_trace` field that satisfies:

- `agent_trace.strategy == 'function_calling'`
- `agent_trace.environment == 'docker'`
- `agent_trace.events` contains at least `model_generate`; whenever the model actually calls a tool, paired `tool_call` / `tool_result` events appear.

### 3.2 Case B: AIME + react + bash + local

The ReAct strategy makes the model reason before acting and lets it call `bash` inside a local subprocess sandbox.

```python
from dotenv import dotenv_values
from evalscope import TaskConfig, run_task
from evalscope.api.agent import AgentConfig

env = dotenv_values('.env')

task_config = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['aime26'],
    dataset_args={'aime26': {'few_shot_num': 0}},
    eval_batch_size=5,
    limit=5,
    generation_config={
        'max_tokens': 2048,
        'temperature': 0.7,
        'parallel_tool_calls': True,
    },
    agent_config=AgentConfig(
        strategy='react',
        tools=['bash'],
        environment='local',
        max_steps=10,
    ),
)

run_task(task_config)
```

Expected outputs:

- `agent_trace.strategy == 'react'`, `agent_trace.environment == 'local'`
- `agent_trace.events` contains at least `model_generate`; paired `tool_call` / `tool_result` events appear when the model invokes a tool, and a single `nudge` may appear when it does not.

```{tip}
- The `local` environment has no filesystem isolation; only use it for development. For production, switch to `docker`.
- `agent_config` also accepts a plain dict — TaskConfig converts it to an `AgentConfig` instance automatically.
```

## SWE-bench Agentic Cases

The `swe_bench_*_agentic` family of benchmarks (`swe_bench_verified_agentic`, `swe_bench_verified_mini_agentic`, `swe_bench_lite_agentic`) is driven directly by `AgentLoopAdapter` subclasses and **does not read `TaskConfig.agent_config`**. All loop parameters are passed via `extra_params` under `dataset_args`.

### 4.1 extra_params reference

| Parameter | Type | Default | Choices | Description |
|-----------|------|---------|---------|-------------|
| `action_protocol` | `str` | `toolcall` | `toolcall` / `backticks` | Agent protocol. `toolcall` reuses the model's native function-calling; `backticks` uses text fenced blocks to support models without function-calling |
| `max_steps` | `int` | `250` | - | Max agent steps per sample |
| `command_timeout` | `float` | `60.0` | - | Default timeout (seconds) per bash command |
| `working_dir` | `str` | `/testbed` | - | Working directory inside the container |
| `build_docker_images` | `bool` | `True` | - | Whether to build per-sample Docker images locally |
| `pull_remote_images_if_available` | `bool` | `True` | - | Whether to try pulling remote images before building |
| `force_arch` | `str` | `''` | `''` / `arm64` / `x86_64` | Force a specific build/pull architecture. Empty string means use the host architecture |

### 4.2 Full example

The following corresponds to the `swe_bench_verified_mini_agentic` configuration in `tests/benchmark/test_agent.py`, evaluating the first 3 samples.

```python
from dotenv import dotenv_values
from evalscope import TaskConfig, run_task

env = dotenv_values('.env')

task_config = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['swe_bench_verified_mini_agentic'],
    dataset_args={
        'swe_bench_verified_mini_agentic': {
            'extra_params': {
                'action_protocol': 'toolcall',
                'max_steps': 250,
                'command_timeout': 60.0,
                'working_dir': '/testbed',
                'build_docker_images': True,
                'pull_remote_images_if_available': True,
            },
        },
    },
    eval_batch_size=2,
    limit=3,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
    },
)

run_task(task_config)
```

```{important}
- Prerequisites: `pip install evalscope[swe_bench]`, and make sure Docker is installed and running.
- If `agent_config` is also configured, **the global config is ignored by swe_bench agentic benchmarks**; tune the loop via `extra_params` instead.
- For sandbox images and remote managers, see [Sandbox Environment](sandbox.md).
- For dataset details, see the [SWE-bench benchmark docs](../third_party/swe_bench.md).
```

## Trace Visualization

When Agent mode is enabled, every sample's evaluation result carries an `agent_trace` field. EvalScope's Web dashboard can replay the full agent trajectory step by step:

1. Start the Web service: `evalscope service --outputs ./outputs`
2. Pick the target report from the dashboard, then go to the **Single Model Detail → Predictions** tab
3. The dashboard detects `agent_trace` and automatically renders an Agent Trace view grouped by `step`

The view shows the following core elements in chronological order:

- **Assistant replies**: model output, reasoning, per-step latency and token usage
- **Tool calls and tool results**: aggregated `tool_call` and the corresponding observations within the same step
- **Environment commands**: actual shell / python_exec commands dispatched into the sandbox
- **Nudge / Error notices**: system reminders when the model did not call a tool, plus parsing errors and tool exceptions
- **Event pills**: compact display of `model_generate / tool_call / tool_result / env_exec / nudge / error / submit` events with elapsed time and token counts

![Agent Trace overview](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/trace_view.png)

```{seealso}
For the full visualization guide (dashboard, report comparison, predictions view, etc.), see [Visualization](../get_started/visualization.md).
```

## FAQ

**Q1: When should I use the global `agent_config` versus benchmark-built-in AgentLoop?**

- Want a regular benchmark (GSM8K, AIME, IFEval, HLE, etc.) to run as a multi-turn tool-calling task → set `TaskConfig.agent_config`.
- Evaluating a benchmark that is intrinsically agentic (SWE-bench agentic, Terminal-Bench, etc.) → just run the dataset and tune `dataset_args.extra_params`; you do **not** need to (and should not) set `agent_config` (it will be ignored).

**Q2: The model never calls any tool. How do I debug?**

1. Verify the tool names in `agent_config.tools` (see section 2.2).
2. Make sure the model supports function-calling. Otherwise switch to a text protocol such as `swe_bench_backticks`.
3. Inspect `agent_trace.events` for `nudge` (model was reminded but still did not call a tool) or `error` (parsing / tool execution failed).
4. Force the model to use a tool via `extra={'system_prompt': '...'}`.

**Q3: What value of `max_steps` should I pick?**

- Math / simple QA benchmarks: `3`–`10` is usually enough.
- Code-repair / SWE-bench style benchmarks: keep the default `250`. A lower value can produce many `max_steps_exceeded` events.

**Q4: I get Docker-related errors with `environment='docker'`. What now?**

- Make sure Docker is installed and running locally (see [Sandbox Environment](sandbox.md)).
- If the first image pull fails, try `docker pull <image>` manually before re-running the evaluation.
- A remote Docker daemon can be reached via `TaskConfig.sandbox.manager_config = {'base_url': 'http://<host>:<port>'}`.
