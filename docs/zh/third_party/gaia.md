# GAIA

## 简介

[GAIA](https://arxiv.org/abs/2311.12983)（General AI Assistants）是一个面向新一代 LLM 的基准，用 450 余道题考察工具使用、网页浏览与多步推理能力。每题都有唯一明确的简短答案（数字、短语，或逗号分隔的列表）。

题目按难度分为三档：

| 难度 | 说明 |
|------|------|
| `2023_level1` | 具备基础工具使用能力的强 LLM 应当能解 |
| `2023_level2` | 需要更自主的规划与更丰富的工具组合 |
| `2023_level3` | agent 能力的明显跃迁——多模态信息源、长链推理 |

每个难度都有公开的 `validation` 划分（带答案）和私有的 `test` 划分（答案保留给[官方 leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)）。EvalScope 当前**仅支持 `validation`**。

GAIA 在 evalscope 的实现为：每个 sample 在独立 Docker 容器里跑一轮 ReAct agent，仅暴露 `bash` 一个工具。打分函数（数字 / 列表 / 字符串归一化）逐字 port 自 GAIA 官方 leaderboard。

## 安装依赖

GAIA 评测在 Docker 中运行：

1. **安装 Docker**：参考 [Docker 安装指南](https://docs.docker.com/engine/install/)，确保守护进程已启动。
2. **安装 evalscope**：
   ```bash
   pip install evalscope
   ```

首次运行会拉取 `python:3.11` 镜像（约 1.1GB），并从 ModelScope 下载 GAIA 数据集快照（约 110MB），都会缓存供后续复用。

```{note}
默认走 [`gaia-benchmark/GAIA`](https://www.modelscope.cn/datasets/gaia-benchmark/GAIA) 的 ModelScope 镜像，无 gating 也无需 token。如需走 HuggingFace 原版，设置 `dataset_hub='huggingface'`（须先在 HF 接受数据集条款）。
```

## 数据集

通过 `subset_list` 在单一 benchmark `gaia` 内切换难度：

| 配置 | 加载内容 |
|------|----------|
| `subset_list=['2023_level1']` | 仅 Level 1 |
| `subset_list=['2023_level1', '2023_level2']` | Level 1 + 2 |
| `subset_list=['2023_level1', '2023_level2', '2023_level3']`（默认）| 全部三档 |

GAIA 约 1/3 题目附带文件（PDF / xlsx / 图片 / 音频 / ...）。EvalScope 会把数据集 `2023/validation/` 目录**只读 mount** 进沙箱的 `/shared_files`，agent 通过 prompt 中嵌入的路径提示访问。

## 运行示例

下例对应 [tests/benchmark/test_agent.py](https://github.com/modelscope/evalscope/blob/main/tests/benchmark/test_agent.py) 中的 `test_gaia`：

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['gaia'],
    dataset_args={
        'gaia': {
            'subset_list': ['2023_level1'],            # 选择运行哪个或哪些难度
            'extra_params': {
                'max_steps': 50,                       # 每个 sample 的 agent loop 上限
                'command_timeout': 180.0,              # 每条 bash 命令的超时（秒）
                'docker_image': 'python:3.11',         # 沙箱镜像；完整版自带 curl/wget/git
                'network_enabled': True,               # 必开：多数题需联网
            }
        }
    },
    eval_batch_size=5,  # 并发沙箱数
    limit=5,            # 快速试跑限制；正式评测请去掉
    generation_config={
        'temperature': 0.7,
        'parallel_tool_calls': True,
        'stream': True,
    }
)
run_task(task_cfg=task_cfg)
```

最终结果示例：

```text
+-----------+---------+----------+-------------+-------+---------+---------+
| Model     | Dataset | Metric   | Subset      |   Num |   Score | Cat.0   |
+===========+=========+==========+=============+=======+=========+=========+
| qwen3-max | gaia    | mean_acc | 2023_level1 |     5 |     0.2 | default |
+-----------+---------+----------+-------------+-------+---------+---------+
```

## 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `max_steps` | int | `50` | 每个 sample 的 agent loop 步数上限。对齐 inspect_ai 的 `message_limit=100`（约 50 个 ReAct 轮次）。 |
| `command_timeout` | float | `180.0` | 每条 bash 命令的超时时间（秒）。 |
| `docker_image` | str | `python:3.11` | 沙箱镜像。完整版 `python:3.11` 自带 `curl` / `wget` / `git`；如需更小可换 `python:3.11-slim`（约 130MB），但需要自己装工具。 |
| `network_enabled` | bool | `True` | 是否允许沙箱联网。GAIA 多数题需要联网。 |

```{note}
GAIA 每个 sample 一个独立 Docker 容器，结束后销毁。`eval_batch_size` 控制同时持有的容器数，请根据机器资源调整。
```

## 评分

评分函数逐字 port 自 [GAIA 官方 leaderboard scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)（Apache 2.0）：

- **数字答案**：去除 `$` / `%` / `,`，按 float 解析，严格相等。
- **列表答案**（含 `,` 或 `;`）：按分隔符切分后逐元素比较（自动判数字 vs 字符串）。
- **字符串答案**：去除空格、小写、去标点，严格相等。

不使用 LLM judge。Agent 必须输出经上述归一化后**严格匹配**的答案——通常通过自动注入的 `submit(answer=...)` 工具调用完成。

## 通过 MCP 接入网页能力

GAIA 默认只有 bash 沙箱，可以 `curl`/`wget` 原始 HTML，但无法跑 JS 渲染浏览器，也接不到需要 API key 的搜索引擎。最简洁的补足方式是接入 MCP 服务器（例如 `mcp-server-fetch` 抓 HTTP，`mcp-server-brave-search` 关键词搜索）—— 这些 MCP server **跑在 host 进程**，不进 docker 沙箱，**镜像不需要改**。

```bash
pip install evalscope[mcp]
```

`evalscope[mcp]` 已经把 `mcp-server-fetch` 一起装进来,下面的示例可以直接跑,无需额外安装。

```python
import sys
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    eval_type='openai_api',
    datasets=['gaia'],
    dataset_args={'gaia': {'subset_list': ['2023_level1']}},
    agent_config=NativeAgentConfig(
        mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                # ``--ignore-robots-txt``：上游 robots.txt 偶尔会请求失败，加上这个
                # 让 fetch 直接尝试目标 URL，避免把所有调用都挡在前面这一步。
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
        ],
    ),
    limit=5,
)
run_task(task_cfg)
```

Agent 此时会看到 `bash`、`submit` 和 `fetch` 三个工具，可以直接 `fetch(url=...)` 抓网页 —— 不需要在 docker 里装额外库，也不再依赖 `curl` 拼凑。

完整配置参考见 [Native Agent Loop → MCP 工具接入](../user_guides/agent/native.md#mcp-工具接入)（stdio / HTTP transport、工具白名单、env 变量等）。

## 已知限制

- **暂无 `web_search` / `web_browser` 工具**：Phase-1 仅提供 `bash`。Agent 可以用 `curl` / `wget` 加 `python3` 解析 HTML，但 JS 渲染页面和需要 API key 的搜索引擎无法直接使用。浏览类题目分数会明显低于具备真实浏览器工具的实现。
- **不支持 `test` 划分**：仅支持 `validation`（有公开答案）。如需向官方 leaderboard 提交，请自行收集模型预测并按 leaderboard 要求格式化。
- **长 bash 输出可能撑爆模型 input 长度**：`curl` 一个大 HTML 页会让消息历史快速膨胀，难度高的样本上偶尔会触发模型最大 input 长度限制。在 `TaskConfig`（或测试中）设置 `ignore_errors=True` 可让评测在遇到此类异常时继续跑剩余样本。
