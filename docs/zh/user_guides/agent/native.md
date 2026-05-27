# 内置 AgentLoop 模式

让常规基准（GSM8K、AIME、IFEval、SWE-bench 等）变成**多轮工具调用**任务，评测模型自身的工具使用与多步推理能力。EvalScope 在每一轮执行 **生成 → 解析 → 调用工具 → 观察 → 再生成**，直到模型提交最终答案或达到步数上限，整个轨迹记录到 `AgentTrace` 用于 [可视化回放](index.md#trace-可视化)。

> 想直接评测 Claude Code / Codex 等成品 Agent CLI，请参见 [外部 Agent Bridge 模式](bridge.md)。

## 何时使用

- **评测模型的多步推理与工具调用能力**
  例：qwen-plus 在 GSM8K 上是否会主动调用 `python_exec` 验证计算，在 AIME 上能否逐步推导并自我检查。

- **让模型在代码修复任务中自主探索**
  例：在 SWE-bench 上让模型通过 `bash` 工具读代码、跑测试、提交补丁，全程无需人工介入。

- **接入 MCP 生态工具，扩展模型能力边界**
  例：给模型挂上 `fetch`、Brave Search 等 MCP 服务器，评测 GAIA 等需要联网检索的 benchmark。

## 快速开始

最小可运行示例：让 qwen-plus 在求解 GSM8K 时通过 `python_exec` 验证计算。

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

EvalScope 自动给模型注入 `python_exec` 工具定义，在 Docker 容器里执行模型生成的 Python 代码，把 stdout/stderr 作为观察回填给模型，直到模型用 `submit` 工具提交答案。

## 常用配置

`NativeAgentConfig` 常用字段：

| 字段 | 说明 | 推荐值 |
|------|------|--------|
| `strategy` | 交互协议 | `function_calling`（默认）；模型不支持时用 `react` 或 `swe_bench_backticks` |
| `tools` | 工具白名单 | 按需，如 `['python_exec']` / `['bash']` |
| `environment` | 工具执行环境 | `local`（开发调试）/ `docker`（生产推荐） |
| `environment_extra` | 沙箱参数 | `docker` 时常用 `{'image': '...', 'timeout': 60}`；详见 [沙箱环境](../sandbox.md) |
| `max_steps` | 单样本最大轮数 | 数学/QA `5-10`，代码修复 `100+` |
| `kwargs` | 策略参数 | 最常用 `{'system_prompt': '...'}` 引导模型行为 |

可选的 `strategy` 值：

| 名称 | 适用场景 |
|------|---------|
| `function_calling`（默认） | 模型支持原生 function-calling |
| `react` | ReAct 风格，让模型先思考再行动 |
| `swe_bench_toolcall` | SWE-bench 工具调用协议 |
| `swe_bench_backticks` | SWE-bench 反引号文本协议（模型不支持 function-calling 时） |

可选的工具（`tools`）：

| 名称 | 说明 |
|------|------|
| `bash` | 执行 shell 命令 |
| `python_exec` | 执行 Python 代码 |
| `submit` | 提交最终答案（自动注入，无需手动配置） |

```{tip}
- `local` 环境无文件系统隔离，生产请改用 `docker`。
- `agent_config` 也接受字典形式，`TaskConfig` 会自动转成 `NativeAgentConfig`。
```

## MCP 工具接入

除了内置工具，`NativeAgentConfig` 通过 `mcp_servers` 接入任意 [MCP（Model Context Protocol）](https://modelcontextprotocol.io) 服务器——把 `fetch`、网页搜索、GitHub 等 MCP 生态能力一键接入评测，不必自己写工具。

### 安装

```bash
pip install evalscope[mcp]
```

`evalscope[mcp]` 自带 `mcp` Python SDK 和 `mcp-server-fetch`（通用 HTTP 抓取，无需 API key）。其他 MCP server 按需安装或通过 `uvx` / `npx` 即调即用。

### 快速开始

给 GAIA 评测挂上 HTTP 抓取和远端 MCP 工具：

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
            # 本地启动 MCP 服务器进程
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
            # 连接远端 MCP endpoint
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

配置好后，agent loop 里模型会同时看到内置工具和所有 MCP 服务器声明的工具，按名称选用。MCP 工具调用如果失败，错误信息会返回给模型让它重试，不会中断整个评测。

### 接入方式

EvalScope 支持三种 MCP 传输协议，按场景选用：

| 配置类 | 适用场景 | 关键参数 |
|--------|---------|---------|
| `MCPServerConfigStdio` | 本地启动 MCP 服务器进程 | `command`、`args`、`env` |
| `MCPServerConfigHTTP` | 连接远端 Streamable HTTP endpoint（推荐） | `url`、`headers` |
| `MCPServerConfigSSE` | 连接旧版 SSE endpoint | `url`、`headers` |

所有配置类共享 `name`（日志显示名）和 `tools`（工具白名单，默认 `'all'`）两个通用字段。完整参数列表见 [源码](https://github.com/modelscope/evalscope/tree/main/evalscope/api/agent/mcp)。

```{tip}
- CI/生产环境推荐 `python -m mcp_server_<x>` + 提前 `pip install`，避免运行时拉包。
- `uvx mcp-server-<x>` 适合临时调试，首次调用会自动安装。
- 远端服务同时提供 `/mcp` 和 `/sse` 时，优先选 `/mcp`（Streamable HTTP）。
```

## 用例：SWE-bench Agentic

`swe_bench_*_agentic` 系列（`swe_bench_verified_agentic`、`swe_bench_verified_mini_agentic`、`swe_bench_lite_agentic`）是自带 AgentLoop 的基准，所有循环参数通过 `dataset_args.extra_params` 传入，**不读取**全局 `agent_config`。

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
                'action_protocol': 'toolcall',  # 或 'backticks'
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

常用 `extra_params`：`action_protocol`、`max_steps`、`command_timeout`、`working_dir`。完整说明见 [SWE-bench 数据集文档](../../third_party/swe_bench.md)。

```{important}
前置依赖：`pip install evalscope[swe_bench]`，本机已装好 Docker。
```

## 常见问题

**模型完全没有调用工具**

1. 检查 `tools` 中的工具名拼写。
2. 确认模型支持 function-calling；不支持的可换 `react` 或 `swe_bench_backticks` 等文本协议。
3. 用 `kwargs={'system_prompt': '...'}` 显式引导模型调用工具。

**`max_steps` 应该设多大**

- 数学/简单 QA：`3`–`10` 通常够用。
- 代码修复/SWE-bench 类：沿用默认 `250`，过低会大量出现 `max_steps_exceeded`。
