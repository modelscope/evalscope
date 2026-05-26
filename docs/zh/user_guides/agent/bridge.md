# 外部 Agent Bridge 模式

让 EvalScope 直接评测 Claude Code、Codex 等成品 Agent CLI。你在 `TaskConfig` 里指定一个评测模型，EvalScope 会在 CLI 与后端模型之间做**协议翻译**(claude-code 发 Anthropic Messages、codex v0.133+ 发 OpenAI Responses，后端模型只需支持 OpenAI Chat Completions 即可)，同时把交互过程录制成 `AgentTrace` 用于 [可视化回放](index.md#trace-可视化)。CLI 本身不需要任何改造。

> 想给 GSM8K / AIME 等常规 benchmark 套上模型自带的多轮工具调用循环，请参见 [内置 AgentLoop 模式](native.md)。

## 何时使用

- **评测成品 Agent CLI 在某 benchmark 上的能力**
  例：Claude Code 在 SWE-bench Pro 上的代码修复得分、Codex 在 GAIA 上的工具调用能力。

- **对比同一 Agent CLI 接到不同后端模型时的表现**
  例：把 claude-code 分别接到 qwen3-max、deepseek-v3、自训模型上跑同一套题，横评后端模型对 Agent 任务的支撑能力。

- **后端模型只懂 OpenAI Chat Completions，但你想驱动 claude-code / codex**
  bridge 在中间自动把 Anthropic Messages / OpenAI Responses 翻译成后端能听懂的 Chat Completions——后端无需原生支持这些协议。

## 快速开始

最小可运行示例:在本地用 Claude Code 跑 GSM8K，模型用 qwen-plus。

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

EvalScope 会在本地子进程里准备 Claude Code(必要时自动 `npm install`)，把它每次发出的 API 请求转给 qwen-plus，结果写到 `outputs/`，并可在 Web UI 按步骤回放 Agent 轨迹。

> 注意这里 `qwen-plus` 走 OpenAI Chat Completions，而 claude-code 发出的是 Anthropic Messages —— bridge 在中间自动完成协议翻译，无需后端原生支持 Anthropic 协议。

## 支持的 Agent CLI

| 名称 | 对应 CLI | 备注 |
|------|---------|------|
| `claude-code` | Anthropic [Claude Code](https://github.com/anthropics/claude-code)(`claude --print`) | 默认推荐 |
| `codex` | OpenAI [Codex](https://github.com/openai/codex)(`codex exec`) | 需要 codex ≥ v0.133 |
| `mock` | 内置 Python 脚本 | 冒烟测试用，无外部依赖 |

想接入 aider、continue 等其他 CLI?见 [进阶:自定义 Runner](#custom-runner)。

## 常用配置

`ExternalAgentConfig` 常用字段:

| 字段 | 说明 | 推荐值 |
|------|------|--------|
| `framework` | 选哪个 CLI | `claude-code` / `codex` |
| `environment` | 跑在哪里 | `local`(开发)/ `docker`(生产) |
| `timeout` | 单样本壁钟超时(秒) | 数学题 120，代码修复 1800+ |
| `environment_extra` | 沙箱构造参数(`image`、`timeout` 等)，与内置模式一致 | 见 [沙箱环境](../sandbox.md) |
| `kwargs` | 透传给 CLI 的参数，见下 | `{}` |

最常调整的 `kwargs` 键(claude-code 与 codex 大体通用，完整列表见 [源码](https://github.com/modelscope/evalscope/tree/main/evalscope/agent/external/runners)):

| 键 | 适用 | 默认 | 何时调整 |
|----|------|------|---------|
| `allowed_tools` | claude-code | 跟随 CLI 默认 | 传空串 `''` 禁用所有工具(适合数学等单轮任务) |
| `auto_install` | 两者 | `True` | 镜像里已预装 CLI，改 `False` 跳过安装 |
| `install_timeout_s` | 两者 | 300(claude-code)/ 600(codex) | 网络慢、首次冷启动超时调大 |
| `home_override` | 两者 | 临时目录 | 需要复用本机 CLI 配置时传一个路径 |

## 用例:SWE-bench Pro 上跑 Claude Code

让 Claude Code 在容器里自主修复代码，样本结束后 EvalScope 自动从工作树取 `git diff` 作为最终补丁。

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

`swe_bench_pro` 会为每个样本自带 Docker 环境，所以 `environment` 留空即可。

## 准备工作

- **`local` 环境**:无额外依赖，主包安装即可。
- **`docker` 环境**:本机已装好并启动 Docker，详见 [沙箱环境](../sandbox.md)。
- **claude-code / codex CLI**:首次运行会在沙箱内自动安装 Node.js + 对应 npm 包，**仅支持 Debian/Ubuntu 系镜像**。

```{tip}
冷启动需要下载 Node 和 npm 包，可能耗时数分钟。生产环境建议把 CLI 预装进镜像并设置 `kwargs={'auto_install': False}`，或为 Docker 挂载持久化 npm 缓存 volume。
```

## 常见问题

**CLI 跑起来了，但 Trace 里看不到任何 `model_generate` 事件**

- **claude-code**:本机如果登录过 Claude OAuth，CLI 会优先读钥匙串，绕过 EvalScope 设的 `ANTHROPIC_BASE_URL`。默认行为下 EvalScope 会用一个临时 `HOME` 规避;如果你显式覆盖了 `home_override`，请确保该目录里没有登录态。
- **codex**:确认版本 ≥ v0.133，旧版本只支持 Chat Completions，与 EvalScope Bridge 不兼容。
- **Docker 场景**:macOS / Windows 的 Docker Desktop 原生提供 `host.docker.internal`;Linux 由 EvalScope 自动注入，通常无需手动配置。

**自动安装失败 / npm 包拉不下来**

- 调大 `install_timeout_s`。
- 或者直接在镜像里预装好 CLI，设置 `kwargs={'auto_install': False}`。
- 长期运行建议挂载 npm 缓存 volume，加速冷启动。

**评测跑得慢 / 单样本超时**

- `timeout` 是单样本上限，SWE-bench 类任务建议 1800–3600 秒。
- `eval_batch_size` 提高并行度(注意宿主机资源)。

(custom-runner)=
## 进阶:自定义 Runner

要接入其他第三方 Agent CLI，需要实现 `AgentRunner` 协议并通过 `@register_runner` 注册。请参考已有实现:

- 协议定义:[runners/base.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/base.py)
- 官方实现:[claude_code.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/claude_code.py)、[codex.py](https://github.com/modelscope/evalscope/blob/main/evalscope/agent/external/runners/codex.py)

注册后即可在 `ExternalAgentConfig(framework='<your-name>')` 中使用。
