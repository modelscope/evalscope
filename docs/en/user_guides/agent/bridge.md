# External Agent Bridge Mode

Evaluate off-the-shelf agent CLIs such as Claude Code, Codex, OpenCode, Gemini CLI, or Hermes directly through EvalScope. You point `TaskConfig` at an evaluation model, and EvalScope sits between the CLI and the backend model as a **protocol translator** (claude-code speaks Anthropic Messages, codex/opencode speak OpenAI Responses, gemini-cli speaks Gemini generateContent, hermes speaks OpenAI Chat Completions, while the backend only needs to support OpenAI Chat Completions). The whole interaction is recorded as an `AgentTrace` for [replay in the UI](index.md#trace-visualization). The CLI itself is untouched.

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

| Name | CLI | Protocol | Notes |
|------|-----|----------|-------|
| `claude-code` | Anthropic [Claude Code](https://github.com/anthropics/claude-code) (`claude --print`) | Anthropic Messages | Recommended default |
| `codex` | OpenAI [Codex](https://github.com/openai/codex) (`codex exec`) | OpenAI Responses | Requires codex ≥ v0.133 |
| `opencode` | [OpenCode](https://github.com/opencode-ai/opencode) (`opencode run`) | OpenAI Responses | Auto-registers model config |
| `gemini-cli` | Google [Gemini CLI](https://github.com/google-gemini/gemini-cli) (`gemini`) | Gemini generateContent | Headless non-interactive mode |
| `hermes` | Nous Research [Hermes Agent](https://github.com/NousResearch/hermes-agent) (`hermes chat`) | OpenAI Chat Completions | Via custom provider config |
| `mock` | bundled Python script | Anthropic Messages | Smoke-test runner; no external dependency |

Want to plug in aider, continue, or another CLI? See [Advanced: custom runners](#custom-runner).

## Common configuration

Most-used `ExternalAgentConfig` fields:

| Field | Description | Recommended |
|-------|-------------|-------------|
| `framework` | Which CLI to drive | `claude-code` / `codex` / `opencode` / `gemini-cli` / `hermes` |
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
- **claude-code / codex / opencode CLIs**: on first run EvalScope auto-installs Node.js + the matching npm package inside the sandbox. **Only Debian/Ubuntu-based images are supported.**
- **gemini-cli**: requires Node.js; use the pre-built image `evalscope-gemini-cli:latest`.
- **hermes**: requires Python 3.11 + uv; use the pre-built image `evalscope-hermes:latest`.

```{tip}
Cold starts download Node and the npm package and can take several minutes. For production, bake the CLI into the image and set `kwargs={'auto_install': False}`, or mount a persistent npm cache volume for Docker.
```

## Pre-built Docker images

EvalScope ships a Dockerfile for each agent CLI under `evalscope/agent/external/dockerfiles/`. Pre-built images skip the runtime installation step and significantly reduce cold-start time.

### Building images

Run from the project root (Docker must be installed):

```bash
# Claude Code
docker build -f evalscope/agent/external/dockerfiles/Dockerfile.claude-code \
             -t evalscope-claude-code:latest .

# Codex
docker build -f evalscope/agent/external/dockerfiles/Dockerfile.codex \
             -t evalscope-codex:latest .

# OpenCode
docker build -f evalscope/agent/external/dockerfiles/Dockerfile.opencode \
             -t evalscope-opencode:latest .

# Gemini CLI
docker build -f evalscope/agent/external/dockerfiles/Dockerfile.gemini-cli \
             -t evalscope-gemini-cli:latest .

# Hermes Agent (clone the source first)
git clone https://github.com/NousResearch/hermes-agent.git \
    evalscope/agent/external/dockerfiles/hermes-agent-src
docker build -f evalscope/agent/external/dockerfiles/Dockerfile.hermes \
             -t evalscope-hermes:latest .
```

### Using images

Set `environment='docker'` in `ExternalAgentConfig` and pass the image name via `environment_extra`:

```python
from evalscope import TaskConfig, run_task
from evalscope.agent.external import ExternalAgentConfig

task_config = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key='<your-key>',
    eval_type='openai_api',
    datasets=['gsm8k'],
    limit=5,
    agent_config=ExternalAgentConfig(
        framework='claude-code',
        environment='docker',
        environment_extra={
            'sandbox_config': {
                'image': 'evalscope-claude-code:latest',
                'network_enabled': True,
            },
        },
        kwargs={
            'auto_install': False,  # CLI is pre-installed, skip setup
        },
    ),
)
run_task(task_config)
```

For other CLIs, just swap `framework` and `image`:

| framework | Recommended image |
|-----------|-------------------|
| `claude-code` | `evalscope-claude-code:latest` |
| `codex` | `evalscope-codex:latest` |
| `opencode` | `evalscope-opencode:latest` |
| `gemini-cli` | `evalscope-gemini-cli:latest` |
| `hermes` | `evalscope-hermes:latest` |

```{note}
- The Dockerfiles include Chinese mirror sources (Aliyun apt/pip/npm) for faster builds in mainland China.
- Customize the base image or add extra dependencies by editing the corresponding Dockerfile.
- `network_enabled: True` allows the container to access the network (some CLIs require it at runtime).
```

## FAQ

**The CLI runs but no `model_generate` events show up in the trace**

- **claude-code**: if your machine is logged into Claude OAuth, the CLI reads the keychain and bypasses the `ANTHROPIC_BASE_URL` EvalScope sets. The default behavior uses a fresh `HOME` to avoid this; if you set `home_override` yourself, make sure the target directory has no stored credentials.
- **codex**: confirm version ≥ v0.133 — older codex only speaks Chat Completions and is incompatible with the bridge.
- **opencode**: the runner auto-writes `~/.config/opencode/opencode.json` to register the model; if the trace is empty, check whether `home_override` points to a directory with an existing config that overrides the bridge endpoint.
- **gemini-cli**: requires `--non-interactive` mode; if the `gemini` command is not found in the container, confirm you are using the pre-built image or have `auto_install=True`.
- **hermes**: the runner injects `provider: custom` + bridge URL via `config.yaml`; if the trace is empty, verify Hermes version ≥ v0.17.
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
- Official runners: [claude_code.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/claude_code.py), [codex.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/codex.py), [opencode.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/opencode.py), [gemini_cli.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/gemini_cli.py), [hermes.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/hermes.py)

Once registered it becomes available as `ExternalAgentConfig(framework='<your-name>')`.
