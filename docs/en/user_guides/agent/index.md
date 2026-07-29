# Agent Evaluation

EvalScope's **Agent Evaluation** lets either the model itself or an off-the-shelf agent CLI complete an evaluation sample inside a controlled multi-turn tool-use loop. The full interaction is recorded for step-by-step replay in the Web UI.

## Two modes

| Mode | When to use | Docs |
|------|-------------|------|
| **Native AgentLoop** | Wrap a regular benchmark (GSM8K, AIME, SWE-bench, …) in a tool-use loop to evaluate the **model's own** tool-use ability | [Native AgentLoop Mode](native.md) |
| **External Agent Bridge** | Evaluate **Claude Code / Codex / OpenCode / Gemini CLI / Hermes** and similar off-the-shelf agent CLIs; EvalScope forwards the CLI's LLM traffic to your evaluation model | [External Agent Bridge Mode](bridge.md) |

Both modes are configured through `TaskConfig.agent_config`. A single evaluation picks exactly one mode.

## Agent environment and task environment

EvalScope separates two different responsibilities:

- `agent_config.environment` selects the Agent execution environment used for command execution and file transfer,
  such as `local` or `docker`; constructor options stay in `agent_config.environment_extra`.
- `agent_config.task_environment` is a stateful task protocol with `reset` / `step` / `state`, plus its service runtime.

For example, MiniWoB uses the `openenv` task backend and hosts its local OpenEnv service with the
`ms_enclave_docker` environment runtime. It does not expose OpenEnv as a shell runtime to AgentLoop tools.

The existing Agent environment interface remains unchanged:
`agent_config.environment` executes the Agent, while
`agent_config.task_environment.runtime` hosts the task-environment service.

For a regular tool-use benchmark, configure the Agent environment:

```json
{"mode": "native", "environment": "docker", "environment_extra": {}}
```

For MiniWoB, leave the Agent environment unset and optionally override its task environment:

```json
{
  "mode": "native",
  "task_environment": {
    "backend": "openenv",
    "observation_mode": "axtree_screenshot",
    "runtime": {"name": "ms_enclave_docker", "config": {}}
  }
}
```

`task_environment.observation_mode` is the common task-environment observation setting; it does not belong in
benchmark `dataset_args`.

## Trace visualization

With Agent Evaluation enabled, every sample carries a trace. The Web UI replays the full interaction step by step:

1. Start the web service: `evalscope service --outputs ./outputs`.
2. Open the dashboard, select the target report and go to **Single-model details → Predictions**.
3. When a trace is detected, the UI renders the Agent Trace view grouped by `step`.

In chronological order the view shows:

- **The model's reply at each step** (reasoning, latency, token usage)
- **Tool calls and their observations** (multiple calls within one step are grouped)
- **Sandbox commands** (bash, python_exec, or the external CLI's actual command line)
- **System reminders and errors** (nudges when the model fails to call a tool, plus parse / tool-execution errors)

![Agent Trace overview](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/trace_view.png)

```{seealso}
For the Web dashboard in general (report comparison, predictions view, …), see [Visualization](../../get_started/visualization.md).
```

## FAQ

**When should I set `agent_config`, and when should I rely on a benchmark's built-in AgentLoop?**

- To turn a regular benchmark (GSM8K, AIME, IFEval, HLE, …) into a multi-turn tool-use task → set [`AgentConfig`](native.md) or [`ExternalAgentConfig`](bridge.md) on `TaskConfig.agent_config`.
- For benchmarks backed by `AgentLoopAdapter` (GAIA, ResearchRubrics, SWE-bench agentic, GDPval, …) → the benchmark provides its required tools and environment, while explicit `NativeAgentConfig` values can override loop settings such as `strategy`, `max_steps`, tools, and MCP servers. Dataset `extra_params` stay benchmark-specific, for example build or filter options.
- Benchmarks like `swe_bench_pro` ship their own per-sample environment and can be combined with the [External Agent Bridge](bridge.md) — just leave `environment` empty.

For mode-specific issues, see each sub-page: [Native AgentLoop FAQ](native.md#faq) · [External Agent Bridge FAQ](bridge.md#faq).

```{toctree}
:hidden:

native.md
bridge.md
```
