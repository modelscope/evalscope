# 内置 AgentLoop 模式

让常规基准(GSM8K、AIME、IFEval、SWE-bench 等)变成**多轮工具调用**任务,评测模型自身的工具使用与多步推理能力。EvalScope 在每一轮执行 **生成 → 解析 → 调用工具 → 观察 → 再生成**,直到模型提交最终答案或达到步数上限,整个轨迹记录到 `AgentTrace` 用于 [可视化回放](index.md#trace-可视化)。

> 想直接评测 Claude Code / Codex 等成品 Agent CLI,请参见 [外部 Agent Bridge 模式](bridge.md)。

## 何时使用

- 评测模型在数学题中是否会正确调用 `python_exec` 验证计算
- 让模型在 ReAct 模式下使用 `bash` 工具自主探索
- 在 SWE-bench 等代码修复基准上测试模型的多步交互能力

## 快速开始

最小可运行示例:让 qwen-plus 在求解 GSM8K 时通过 `python_exec` 验证计算。

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

EvalScope 自动给模型注入 `python_exec` 工具定义,在 Docker 容器里执行模型生成的 Python 代码,把 stdout/stderr 作为观察回填给模型,直到模型用 `submit` 工具提交答案。

## 三个核心组件

每次 AgentLoop 都由这三块拼成,在 `AgentConfig` 中分别配置:

**Strategy** — 控制模型与循环的交互方式:

| 名称 | 适用 |
|------|------|
| `function_calling`(默认) | 模型支持原生 function-calling 时使用 |
| `react` | ReAct 风格,推理优先,适合让模型先思考再行动 |
| `swe_bench_toolcall` | SWE-bench 工具调用协议(模型支持 function-calling) |
| `swe_bench_backticks` | SWE-bench 反引号文本协议(模型不支持 function-calling) |

**Tools** — 模型可用的能力白名单:

| 名称 | 说明 |
|------|------|
| `bash` | 执行 shell 命令 |
| `python_exec` | 执行 Python 代码 |
| `submit` | 提交最终答案(`function_calling` / `react` 自动注入,无需手动配置) |

**Environment** — 工具实际跑在哪:

| 名称 | 说明 |
|------|------|
| `local` | 本地子进程,无文件系统隔离,**仅推荐开发调试** |
| `docker` | 基于 [ms-enclave](https://github.com/modelscope/ms-enclave) 的容器,生产推荐 |

## 常用配置

`AgentConfig` 常用字段:

| 字段 | 说明 | 推荐值 |
|------|------|--------|
| `strategy` | 选哪种交互协议 | `function_calling`(默认) |
| `tools` | 工具白名单 | 按需,如 `['python_exec']` / `['bash']` |
| `environment` | 工具执行环境 | `local`(开发)/ `docker`(生产) |
| `environment_extra` | 沙箱构造参数 | `docker` 时常用 `{'image': '...', 'timeout': 60}`;详见 [沙箱环境](../sandbox.md) |
| `max_steps` | 单样本最大轮数 | 数学/QA `5-10`,代码修复 `100+` |
| `extra` | 策略相关参数 | 最常用 `{'system_prompt': '...'}` 引导模型 |

```{tip}
- `local` 环境无隔离,生产请改用 `docker`。
- `agent_config` 也接受字典形式,`TaskConfig` 会自动转成 `AgentConfig`。
- 启用 `docker` 时复用 [沙箱基础设施](../sandbox.md),可显式 `TaskConfig.sandbox = SandboxTaskConfig(enabled=True, engine='docker')`。
```

## MCP 工具接入

除了内置工具（`bash` / `python_exec` 等），`AgentConfig` 通过 `mcp_servers` 接受任意 [MCP（Model Context Protocol）](https://modelcontextprotocol.io)服务器。它们暴露的工具会与 benchmark 自带工具合并进 agent loop —— **benchmark 代码不需要任何改动**。

这是把 `fetch`、网页搜索、GitHub、文件系统等 MCP 生态一键接入 EvalScope 的方式，不必自己写 EvalScope 工具。

### 安装

```bash
pip install evalscope[mcp]
```

`evalscope[mcp]` 自带官方 `mcp` Python SDK 加 `mcp-server-fetch`(通用 HTTP 抓取,无需 API key)。其他 MCP server —— 比如 `mcp-server-brave-search`、`mcp-server-github`、`mcp-server-filesystem`、`mcp-server-puppeteer` —— 按需 `pip install <name>` 加装,或通过 `uvx <name>` / `npx <name>` 即调即用。

`mcp` Python SDK 是延迟导入的,运行时 `mcp_servers` 为空(默认)的配置不需要额外开销。

### 快速开始

```python
import sys
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio, MCPServerConfigHTTP, MCPServerConfigSSE

task_config = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    eval_type='openai_api',
    datasets=['gaia'],
    agent_config=NativeAgentConfig(
        mcp_servers=[
            # Stdio:在本地启动一个 Python MCP 服务器
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
            # Stdio:用 uvx 让最终用户不必先 pip install
            MCPServerConfigStdio(
                command='uvx',
                args=['mcp-server-brave-search'],
                env={'BRAVE_API_KEY': 'sk-...'},
                name='brave_search',
            ),
            # Streamable HTTP:连接远端 MCP endpoint(推荐)
            # 例:ModelScope MCP 广场,URL 末尾通常是 /mcp
            MCPServerConfigHTTP(
                url='https://mcp.api-inference.modelscope.net/<server-id>/mcp',
                headers={'Authorization': 'Bearer <MODELSCOPE_SDK_TOKEN>'},
                name='ms_fetch',
            ),
            # SSE:旧版 MCP HTTP transport,URL 末尾通常是 /sse
            MCPServerConfigSSE(
                url='https://mcp.api-inference.modelscope.net/<server-id>/sse',
                headers={'Authorization': 'Bearer <MODELSCOPE_SDK_TOKEN>'},
                name='ms_fetch_sse',
            ),
        ],
    ),
    dataset_args={'gaia': {'subset_list': ['2023_level1']}},
    limit=5,
)
run_task(task_config)
```

```{tip}
**Streamable HTTP vs SSE**:Streamable HTTP 是新版 MCP HTTP transport,SSE 是旧版。两个 endpoint 功能等价,服务方同时提供时优先选 `/mcp`(Streamable HTTP)。`MCPServerConfigSSE` 主要用于只提供 `/sse` 的老服务。
```

agent loop 里模型现在会同时看到 `bash`、`submit` **以及** 各 MCP 服务器声明的全部工具(如 `fetch`、`brave_web_search`),按名称选用。

### 参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `command`(stdio) | str | 启动可执行文件(`sys.executable` / `uvx` / `npx` / 绝对路径) |
| `args`(stdio) | list[str] | 传给 `command` 的参数 |
| `env`(stdio) | dict[str, str] | 子进程环境变量 |
| `cwd`(stdio) | str | 子进程工作目录 |
| `url`(http / sse) | str | MCP server 的 Streamable HTTP endpoint(`http`)或 SSE endpoint(`sse`) |
| `headers`(http / sse) | dict[str, str] | HTTP headers(通常用于鉴权) |
| `timeout`(http) | float | HTTP 读超时秒数(默认 `30.0`) |
| `timeout`(sse) | float | 非流式 HTTP 操作超时秒数(默认 `5.0`) |
| `sse_read_timeout`(sse) | float | 等待下一个 SSE 事件的超时秒数,超时即断开(默认 `300.0`) |
| `name` | str | 日志和 trace UI 中显示的名字。不填则用 `command` / `url` |
| `tools` | `'all'` 或 list[str] | 工具白名单;`'all'`(默认)暴露 server 声明的全部工具 |

```{tip}
- CI/测试推荐 `python -m mcp_server_<x>` + 提前 `pip install`,每次跑零额外开销。
- `uvx mcp-server-<x>` 适合 ad-hoc 调试,首次调用会拉包。
- 即使 benchmark 自己驱动 agent loop(如 GAIA / SWE-bench-agentic 这类 `AgentLoopAdapter` 子类),全局 `agent_config.mcp_servers` 仍会被读取并 merge 进它们的工具集。
```

### Trace / 调试

每个 MCP server 启动和关闭时会在日志里打 `MCPServer[<name>]: initialised` / `closed`。MCP 工具调用在 `agent_trace.events` 里和原生工具一样以 `tool_call` / `tool_result` 形式出现，`name` 就是 MCP 声明的工具名。

MCP 工具调用如果失败,observation 会以 `[error] <错误内容>` 的形式返回模型,模型可以据此换个方式重试,不会让整个 loop 崩。

## SWE-bench Agentic 基准

`swe_bench_*_agentic` 系列(`swe_bench_verified_agentic`、`swe_bench_verified_mini_agentic`、`swe_bench_lite_agentic`)是自带 AgentLoop 的基准,**不读取**全局 `agent_config`,所有循环参数通过 `dataset_args.extra_params` 传入。

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
                'action_protocol': 'toolcall',  # 或 'backticks'(模型不支持 function-calling 时)
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

常用 `extra_params`:`action_protocol`、`max_steps`、`command_timeout`、`working_dir`、`build_docker_images`、`pull_remote_images_if_available`、`force_arch`。完整说明见 [SWE-bench 数据集文档](../../third_party/swe_bench.md)。

```{important}
前置依赖:`pip install evalscope[swe_bench]`,本机已装好 Docker。同时配置了 `agent_config` 也会被**忽略**。
```

## 常见问题

**模型完全没有调用工具**

1. 检查 `tools` 中的工具名拼写。
2. 确认模型支持 function-calling;不支持的可换 `swe_bench_backticks` 等文本协议。
3. 查看 `agent_trace.events` 是否出现 `nudge`(系统已提醒模型调用工具)或 `error`(解析 / 执行报错)。
4. 用 `kwargs={'system_prompt': '...'}` 显式要求模型调用某个工具。

**`max_steps` 应该设多大**

- 数学 / 简单 QA:`3`–`10` 通常够用。
- 代码修复 / SWE-bench 类:沿用默认 `250`,过低会大量出现 `max_steps_exceeded`。
