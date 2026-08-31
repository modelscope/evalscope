# Toolathlon Official Service Wrapper


## 概述

Toolathlon 是一个面向现实场景、长周期工具使用的智能体基准测试，覆盖多种基于 MCP 的软件环境。本 EvalScope 基准测试是对官方 Toolathlon 远程评估服务的封装，而非对 MCP 环境或官方评估器的本地重新实现。

## 任务描述

- **任务类型**：长周期智能体工具使用，支持多轮函数调用
- **输入**：从 Toolathlon-Verified 任务集中选取的真实软件任务
- **输出**：由官方 Toolathlon 智能体循环执行的模型响应和工具调用
- **领域**：基于 MCP 的生产力、开发、数据及 Web 软件环境

## 核心特性

- 使用官方 Toolathlon 服务进行 MCP 环境、任务容器、智能体执行和评分
- 支持官方服务的私有模式（private mode），可搭配本地或内网的 OpenAI 兼容模型端点
- 将一次远程 Toolathlon 作业视为一个 EvalScope 样本，通过 `task_list` 和 `limit` 参数选择作业中的具体任务
- 对临时性的轮询和结果下载失败自动重试，而不取消远程评估
- 对失败的归档下载保持挂起状态以供后续重试，并验证已下载的输出路径
- 模型 API 密钥保留在 EvalScope 侧，并通过环境变量传递给本地 WebSocket 中继
- 包含从官方仓库提交 `b7bbac3f9a1f381b095c878debe1a47dd164ad85` 中提取的 Toolathlon-Verified 任务列表

## 评估说明

- 主要指标为官方 Toolathlon 评分器报告的准确率（accuracy）
- 聚合分数从官方统计中读取，包括 `average_success_rate`，并为兼容的服务输出提供计数和按任务回退机制
- 缺失、非有限值或超出范围的分数将导致验证失败，而非静默报告为零
- EvalScope 评估语义版本为 `v1.1`；本封装遵循 Toolathlon 服务协议 `1.3`
- 运行时依赖 `httpx`、`websockets`，且 EvalScope 进程需能访问 OpenAI 兼容端点
- 共享公共服务存在队列和基于 IP 的使用限制；如需持续评估，请使用专用或自托管服务

## 使用指南

有关公共服务限制、私有模式数据流、自托管服务设置以及 EvalScope 配置示例，请参阅 Toolathlon 使用指南：

- https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolathlon.html

官方资源：

- https://github.com/hkust-nlp/Toolathlon
- https://github.com/hkust-nlp/Toolathlon/blob/main/EVAL_SERVICE_README.md


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `toolathlon` |
| **数据集ID** | [Toolathlon](https://github.com/hkust-nlp/Toolathlon) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1 |
| 提示词长度（平均） | 50 字符 |
| 提示词长度（最小/最大） | 50 / 50 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "ba8702fc",
      "content": "Run Toolathlon official remote evaluation service."
    }
  ],
  "target": "",
  "id": 0,
  "metadata": {
    "task_list": [
      "ab-testing",
      "academic-pdf-report",
      "academic-warning",
      "add-bibtex",
      "apply-phd-email",
      "arrange-workspace",
      "canvas-arrange-exam",
      "canvas-art-manager",
      "canvas-art-quiz",
      "canvas-do-quiz",
      "... [TRUNCATED 98 more items] ..."
    ],
    "mode": "private"
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板:**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `mode` | `str` | `private` | Toolathlon 服务模式。EvalScope 此封装仅支持私有模式。选项：['private'] |
| `server_host` | `str` | `47.253.6.47` | 官方 Toolathlon 评估服务主机地址。 |
| `server_port` | `int` | `8080` | 官方 Toolathlon HTTP 服务端口。 |
| `ws_proxy_port` | `int` | `8081` | 私有模式下官方 Toolathlon WebSocket 代理端口。 |
| `workers` | `int` | `10` | 向官方服务请求的并行 Toolathlon 工作进程数量。 |
| `provider` | `str` | `unified` | Toolathlon 模型提供者类型。选项：['unified', 'openai_stateful_responses'] |
| `task_list` | `list` | `[]` | 可选的 Toolathlon 任务名称列表。留空则使用内置的 Toolathlon-Verified 列表。 |
| `task_list_file` | `str` | `` | 可选文件路径，每行包含一个 Toolathlon 任务名称。 |
| `model_params` | `dict` | `{}` | 额外模型参数，将转发给 Toolathlon，在 TaskConfig.generation_config 之后合并。 |
| `job_id` | `str` | `` | 可选的 Toolathlon 作业 ID。可用于恢复未完成的官方服务作业。 |
| `force_redownload` | `bool` | `False` | 强制重新下载 Toolathlon 结果归档。 |
| `override_output_dir` | `bool` | `False` | 当 Toolathlon 输出目录已存在文件时，清空该目录。 |
| `skip_container_restart` | `bool` | `False` | 跳过 Toolathlon 容器重启。仅用于小型调试任务子集。 |
| `trust_env_in_httpx` | `bool` | `False` | 允许 httpx 使用代理环境变量。 |
| `timeout_seconds` | `int` | `14400` | 等待 Toolathlon 官方服务作业的最大时间（秒）。 |
| `poll_interval` | `int` | `5` | 轮询 Toolathlon 作业状态的时间间隔（秒）。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets toolathlon \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['toolathlon'],
    dataset_args={
        'toolathlon': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
