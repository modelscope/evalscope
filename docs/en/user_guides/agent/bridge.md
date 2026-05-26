# External Agent Bridge Mode

Evaluate off-the-shelf agent CLIs such as Claude Code or Codex directly through EvalScope. You point `TaskConfig` at an evaluation model, and EvalScope sits between the CLI and the backend model as a **protocol translator** (claude-code speaks Anthropic Messages, codex v0.133+ speaks OpenAI Responses, while the backend only needs to support OpenAI Chat Completions). The whole interaction is recorded as an `AgentTrace` for [replay in the UI](index.md#trace-visualization). The CLI itself is untouched.

> To wrap GSM8K / AIME and other regular benchmarks in the model's own multi-turn tool-use loop, see [Native AgentLoop Mode](native.md).

## When to use

- **Benchmark an off-the-shelf agent CLI on a specific dataset.**
  e.g. Claude Code's code-fix score on SWE-bench Pro, Codex's tool-use ability on GAIA.

- **Compare the same agent CLI across different backend models.**
  e.g. point claude-code at qwen3-max, deepseek-v3, your own fine-tuned model on the same task set to see which backend best supports agentic workloads.

- **Your backend only speaks OpenAI Chat Completions, but you still want to drive claude-code / codex.**
  The bridge transparently translates Anthropic Messages / OpenAI Responses into Chat Completions on the way out — the backend never needs to support those protocols natively.

## Quick start

Smallest runnable example — Claude Code running locally against GSM8K with `qwen-plus`.

```python
from evalscope import TaskConfig, run_task
from evalscope.agent.external import ExternalAgentConfig

task_config = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key='<your-key>',
    eval_type='openai_api',
    datasets=['gsm8k'],
    limit=3,
    agent_config=ExternalAgentConfig(
        framework='claude-code',
        environment='local',
    ),
)
run_task(task_config)
```

EvalScope prepares Claude Code in a local subprocess (auto `npm install` if needed), routes its API requests to `qwen-plus`, writes the results to `outputs/`, and lets you replay the agent trajectory step by step in the Web UI.

> Note that `qwen-plus` speaks OpenAI Chat Completions while claude-code emits Anthropic Messages — the bridge translates between them transparently, so the backend never needs to support the Anthropic protocol natively.

## Supported agent CLIs

| Name | CLI | Notes |
|------|-----|-------|
| `claude-code` | Anthropic [Claude Code](https://github.com/anthropics/claude-code) (`claude --print`) | Recommended default |
| `codex` | OpenAI [Codex](https://github.com/openai/codex) (`codex exec`) | Requires codex ≥ v0.133 |
| `mock` | bundled Python script | Smoke-test runner; no external dependency |

Want to plug in aider, continue, or another CLI? See [Advanced: custom runners](#custom-runner).

## Common configuration

Most-used `ExternalAgentConfig` fields:

| Field | Description | Recommended |
|-------|-------------|-------------|
| `framework` | Which CLI to drive | `claude-code` / `codex` |
| `environment` | Where to run | `local` (dev) / `docker` (production) |
| `timeout` | Per-sample wall-clock budget in seconds | 120 for math, 1800+ for code fixes |
| `environment_extra` | Sandbox constructor kwargs (`image`, `timeout`, …); same as native mode | See [Sandbox Environment](../sandbox.md) |
| `kwargs` | Kwargs forwarded to the CLI, see below | `{}` |

Most-tweaked `kwargs` keys (largely shared between claude-code and codex; full list in [source](https://github.com/modelscope/evalscope/tree/main/evalscope/agent/external/runners)):

| Key | Applies to | Default | When to change |
|-----|-----------|---------|----------------|
| `allowed_tools` | claude-code | CLI default | Empty string `''` disables every tool (good for math / single-turn tasks) |
| `auto_install` | both | `True` | Set `False` when the CLI is already baked into the image |
| `install_timeout_s` | both | 300 (claude-code) / 600 (codex) | Bump on slow networks / cold-start timeouts |
| `home_override` | both | tempdir | Set to a path when you need to reuse the host's CLI configuration |

## Example: Claude Code on SWE-bench Pro

Let Claude Code autonomously fix code inside a container; when the sample finishes EvalScope automatically takes `git diff` from the working tree as the final patch.

```python
task_config = TaskConfig(
    model='qwen-plus',
    api_url='...',
    api_key='...',
    eval_type='openai_api',
    datasets=['swe_bench_pro'],
    limit=3,
    agent_config=ExternalAgentConfig(
        framework='claude-code',
        timeout=1800.0,
    ),
)
run_task(task_config)
```

`swe_bench_pro` ships its own per-sample Docker environment, so leaving `environment` empty is fine.

## Prerequisites

- **`local` environment**: no extra dependencies — the main package is enough.
- **`docker` environment**: Docker installed and running locally; see [Sandbox Environment](../sandbox.md).
- **claude-code / codex CLIs**: on first run EvalScope auto-installs Node.js + the matching npm package inside the sandbox. **Only Debian/Ubuntu-based images are supported.**

```{tip}
Cold starts download Node and the npm package and can take several minutes. For production, bake the CLI into the image and set `kwargs={'auto_install': False}`, or mount a persistent npm cache volume for Docker.
```

## FAQ

**The CLI runs but no `model_generate` events show up in the trace**

- **claude-code**: if your machine is logged into Claude OAuth, the CLI reads the keychain and bypasses the `ANTHROPIC_BASE_URL` EvalScope sets. The default behavior uses a fresh `HOME` to avoid this; if you set `home_override` yourself, make sure the target directory has no stored credentials.
- **codex**: confirm version ≥ v0.133 — older codex only speaks Chat Completions and is incompatible with the bridge.
- **Docker scenarios**: Docker Desktop on macOS / Windows provides `host.docker.internal` natively; on Linux EvalScope injects it automatically. Usually no manual setup needed.

**Auto-install fails / npm package can't be pulled**

- Bump `install_timeout_s`.
- Or bake the CLI into the image and set `kwargs={'auto_install': False}`.
- For long-running pipelines, mount an npm cache volume to speed up cold starts.

**Evaluation is slow / per-sample timeout**

- `timeout` is the per-sample cap; SWE-bench-style tasks need 1800–3600 seconds.
- Raise `eval_batch_size` for more parallelism (mind host resources).

(custom-runner)=
## Advanced: custom runners

To plug in a third-party agent CLI, implement the `AgentRunner` protocol and register it with `@register_runner`. Reference the existing implementations:

- Protocol: [runners/base.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/base.py)
- Official runners: [claude_code.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/claude_code.py), [codex.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/codex.py)

Once registered it becomes available as `ExternalAgentConfig(framework='<your-name>')`.
