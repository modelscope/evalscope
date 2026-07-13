# WideSearch

## 简介

[WideSearch](https://github.com/ByteDance-Seed/WideSearch) 用于评估搜索 Agent 能否从互联网收集大量且完整的事实，
并按照指定结构输出 Markdown 表格。ModelScope 数据集
[`bytedance-community/WideSearch`](https://modelscope.cn/datasets/bytedance-community/WideSearch) 的 `full` split
包含 200 道题，其中英文和中文各 100 道；每道题都带有 gold CSV 和逐列评分配置。

EvalScope 接入的是官方 single-agent 设置，使用官方中英文 system prompt、`function_calling`、默认 bash 工具，
并严格移植官方表格对齐及规则/LLM 混合评分流程。本接入不包含官方 `create_sub_agents` multi-agent 基线。

## 安装

```bash
pip install 'evalscope[wide_search]'
```

根据运行方式安装可选依赖：

```bash
pip install 'evalscope[wide_search,mcp]'      # Fetch MCP 及其他 MCP server
pip install 'evalscope[wide_search,sandbox]'  # Docker sandbox
```

## 默认本地评测

默认环境是每题独立的临时本地目录，能够使用主机网络，并会在题目完成后清理。该环境不是安全沙箱：模型仍可通过
绝对路径访问主机文件，因此只应运行可信模型。

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    api_url='OPENAI_COMPATIBLE_URL',
    api_key=os.getenv('MODEL_API_KEY'),
    eval_type='openai_api',
    datasets=['wide_search'],
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_JUDGE_MODEL',
        'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
        'api_key': os.getenv('JUDGE_API_KEY'),
        'generation_config': {'temperature': 0.0},
    },
    eval_batch_size=1,
    limit=1,  # 正式评测 200 题时删除
))
```

默认最多运行 50 个 AgentLoop step，每次 bash 命令超时为 120 秒。可通过
`NativeAgentConfig(max_steps=..., command_timeout=...)` 覆盖。

## Docker 环境

通过 EvalScope 统一的 sandbox 配置启用 Docker。默认镜像为 `python:3.11-slim`，并允许网络访问。

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.config import SandboxTaskConfig

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['wide_search'],
    agent_config=NativeAgentConfig(max_steps=50, command_timeout=120),
    sandbox=SandboxTaskConfig(
        enabled=True,
        default_config={
            'image': 'python:3.11-slim',
            'network_enabled': True,
        },
    ),
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
    limit=1,
))
```

Docker 模式要求 Docker daemon 正常运行，并安装 `evalscope[sandbox]`。

## Fetch MCP 与论文 repeats 配置

MCP 不是必需依赖。下面的配置保留 bash，同时加入官方 Fetch MCP server，并为每道题运行四次，以生成论文形式的
`Avg@4`、`Pass@4` 和 `Max@4` 报告。

```python
import sys

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['wide_search'],
    repeats=4,
    agent_config=NativeAgentConfig(
        max_steps=50,
        command_timeout=120,
        mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            )
        ],
    ),
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```

## 评分与报告

评分器会解析 Markdown 表格、归一化列名、使用 judge 对齐语义等价的列名和主键实体、连接预测与 gold 行，并按数据集
配置执行预处理和指标计算。支持的官方操作包括 `norm_str`、`extract_number`、`norm_date`、`exact_match`、
`number_near`、`date_near`、`url_match` 和 `llm_judge`。

每次 trial 输出七项指标：

- `success_rate`
- `row_precision`、`row_recall`、`row_f1`
- `item_precision`、`item_recall`、`item_f1`

一次 full 运行会在不重复推理的情况下派生 `all`、`en`、`zh` 三组报告。Success Rate 输出 `Avg@N` 和
`Pass@N`，Row/Item 指标输出 `Avg@N` 和 `Max@N`。使用 `repeats=4` 可得到论文形式的报告结构。

列名和实体对齐同样依赖 judge，因此本 benchmark 不支持 rule-only 评分。论文推荐使用
GPT-4.1-2025-04-14 以获得可比结果，但 EvalScope 不会硬编码 judge 模型。

## 相关资源

- [论文](https://arxiv.org/abs/2508.07999)
- [官方代码仓库](https://github.com/ByteDance-Seed/WideSearch)
- [ModelScope 数据集](https://modelscope.cn/datasets/bytedance-community/WideSearch)
