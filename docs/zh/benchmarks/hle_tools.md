# Humanity's-Last-Exam-with-Tools

## 概述

Humanity's Last Exam with Tools（`hle_tools`）是闭卷评测集 [`hle`](hle.md) 的工具使用版本。它复用 CAIS / Scale AI 的同一批 2,500 道专家级题目，但通过 EvalScope Native AgentLoop 驱动每条样本，使模型在作答前可以调用代码执行和可选的网页工具。

这是与 `hle` **不同的排行榜条目**。闭卷问答请使用 `datasets=['hle']`，多轮工具使用协议请使用 `datasets=['hle_tools']`。

## 任务描述

- **任务类型**：带工具的专家级问答（多轮 AgentLoop）
- **输入**：问题（14% 为多模态，含图片），并提供 `python_exec`（以及可选的 MCP fetch/search）
- **输出**：答案、解释及置信度分数
- **领域分布**：数学（41%）、物理（9%）、生物/医学（11%）、计算机科学/AI（10%）、人文（9%）、工程（4%）、化学（7%）、其他（9%）

## 主要特点

- 复用官方 HLE 数据集（ModelScope 上的 `cais/hle`）、裁判（`GRADE: C/I`）、子集以及 `include_multi_modal` 额外参数
- 默认 Native AgentLoop：`function_calling` 策略、`python_exec`、`local` 环境、最多 30 步
- 安装 `evalscope[mcp]`（及 `mcp-server-fetch`）后自动挂载可选 MCP `fetch`，无需付费搜索 API Key
- 可通过 `NativeAgentConfig.mcp_servers` 额外接入 MCP 网页搜索（Brave、Tavily 等）
- 正式评测推荐使用 Docker 隔离，但**不是**必须；默认 `local` 环境便于本地/Mock 验证

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：与 HLE 相同的 LLM 裁判 **准确率（Accuracy）**
- 响应格式包括：解释（Explanation）、答案（Answer）和置信度（Confidence，0–100%）
- **注意**：对于纯文本模型，请将 `extra_params["include_multi_modal"]` 设为 `False`
- 使用 GRADE: C/I 格式进行 LLM 裁判评分
- `datasets=['hle_tools']` 默认启用工具。除非需要覆盖循环配置，否则不必设置 `TaskConfig.agent_config`
- 传入 `NativeAgentConfig` 时，未显式设置的字段（`tools`、`environment`、`max_steps`、`mcp_servers`）会与默认值合并
- `local` 没有文件系统隔离；正式评测请设置 `environment='docker'`（见下文）
- 网页抓取请安装 `pip install evalscope[mcp]`。付费搜索 MCP 为可选项

## Agent 环境

默认循环（未设置 `agent_config` 时生效）：

- **策略**：`function_calling`
- **工具**：内置 `python_exec`。`submit` 会自动注入
- **环境**：`local`（宿主机子进程）。正式评测推荐覆盖为 `docker` + `python:3.11-slim`
- **MCP**：安装可选依赖后使用 `mcp-server-fetch`
- **max_steps**：30

覆盖示例 — Docker + fetch，以及可选的搜索服务器：

```python
import sys
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['hle_tools'],
    dataset_args={
        'hle_tools': {
            'subset_list': ['Math'],
            'extra_params': {'include_multi_modal': False},
        }
    },
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        tools=['python_exec'],
        environment='docker',
        environment_extra={'sandbox_config': {'image': 'python:3.11-slim'}},
        max_steps=30,
        mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
            # 可选付费搜索，例如 Brave / Tavily MCP — 非必须
        ],
    ),
    limit=10,
))
```

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hle_tools` |
| **数据集ID** | [cais/hle](https://modelscope.cn/datasets/cais/hle/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2501.14249) |
| **标签** | `Agent`, `Knowledge`, `MultiTurn`, `QA` |
| **指标** | `accuracy` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,500 |
| 提示词长度（平均） | 1029.85 字符 |
| 提示词长度（最小/最大） | 234 / 21341 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Biology/Medicine` | 280 | 1259.39 | 246 | 13702 |
| `Chemistry` | 165 | 812.72 | 236 | 6942 |
| `Computer Science/AI` | 241 | 1581.02 | 263 | 11529 |
| `Engineering` | 111 | 1620.26 | 250 | 21341 |
| `Humanities/Social Science` | 219 | 1069.39 | 256 | 7028 |
| `Math` | 1,021 | 862.46 | 262 | 8952 |
| `Physics` | 230 | 1027.63 | 257 | 17139 |
| `Other` | 233 | 754.94 | 234 | 13655 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 342 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 329x12 – 14950x2780 |
| 图像格式 | gif, jpeg, png, webp |

## 样例示例

**子集**: `Biology/Medicine`

```json
{
  "input": [
    {
      "id": "906a518f",
      "content": "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"
    },
    {
      "id": "d03d8d4e",
      "content": [
        {
          "text": "In a bioinformatics lab, Watterson's estimator (theta) and pi (nucleotide diversity) will be calculated from variant call files which contain human phased samples with only single nucleotide variants present, and there are no completely missi ... [TRUNCATED] ... y pi (nucleotide diversity) is biased.\nC. Both Watterson's estimator (theta) and pi (nucleotide diversity) are biased.\nD. Neither Watterson's estimator (theta) nor pi (nucleotide diversity) are biased.\nE. None of the other answers are correct"
        }
      ]
    }
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "Biology/Medicine",
  "metadata": {
    "uid": "66e88728ba7d8bc0d5806f3a",
    "author_name": "Scott S",
    "rationale": "First, we recognize that all single nucleotide variants are included somewhere in the sample. It is given that, across “all samples,” there are no “missing single nucleotide variants.” Further, since “[t]he number of samples is arbitrarily la ... [TRUNCATED] ... fferent genotypes that that position, the analysis would consider these two genomes to have the same nucleotide at the position. This reduces the estimated nucleotide diversity, pi. Therefore, pi would be biased in the circumstance described.",
    "raw_subject": "Bioinformatics",
    "category": "Biology/Medicine",
    "has_image": false
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `include_multi_modal` | `bool` | `True` | 评估时是否包含多模态（图像）问题。 |

## 使用方法

### 通过命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hle_tools \
    --limit 10  # 正式评估时请删除此行
```

### 通过 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['hle_tools'],
    dataset_args={
        'hle_tools': {
            # subset_list: ['Biology/Medicine', 'Chemistry', 'Computer Science/AI']  # 可选，用于评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
