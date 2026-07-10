# WideSearch


## 概述

WideSearch 评估搜索智能体在广泛信息检索任务上的表现：从实时网络中收集大量、完整的原子事实，并将其组织成一个严格结构化的 Markdown 表格。该基准测试包含 200 个手动整理的任务，英文和中文任务各占一半。

## 任务与运行时

- **输入**：带有明确 Markdown 表格 schema 的自然语言收集请求
- **输出**：一个完整的 Markdown 表格
- **智能体**：EvalScope 基准测试自带的 AgentLoop，使用官方单智能体提示词和函数调用策略
- **工具**：默认使用 Bash；可选的 MCP 服务器通过 ``NativeAgentConfig.mcp_servers`` 合并
- **环境**：默认为每个样本创建临时本地目录，也可选择 Docker 沙箱

本地环境使用主机网络，不具备安全沙箱特性：绝对路径仍可访问主机文件。仅在使用可信模型时才应启用此模式。如需隔离，请设置 ``environment_type='docker'``，或附加搜索和页面抓取 MCP 服务器以构建更具代表性的搜索智能体配置。本适配器未实现官方多智能体 ``create_sub_agents`` 基线。

## 评估

评分器保留了官方的表格对齐方式和混合单元格级评估流程。它会解析 Markdown 表格，通过 LLM 评判器对齐列名和主键实体，然后对每列分别应用精确匹配、数值匹配、日期匹配、URL 匹配或语义匹配。不支持纯规则评判；请提供 ``judge_model_args`` 并设置 ``judge_strategy='auto'`` 或 ``'llm'``。建议使用 GPT-4.1-2025-04-14 以与论文结果进行比较。

每次试验报告表格成功率以及行/项级别的精确率、召回率和 F1 分数。重复试验结果将聚合为论文中的 Avg@N、Pass@N 和 Max@N 指标，分别针对 ``all``、``en`` 和 ``zh``，且不会对同一样本重复评估。使用 ``repeats=4`` 可复现论文的报告形式。

## 配置

- ``max_steps``：默认为 50
- ``environment_type``：``local``（默认）或 ``docker``
- ``command_timeout``：Bash 命令超时时间为 120 秒
- ``docker_image``：``python:3.11-slim``
- ``network_enabled``：Docker 环境默认启用网络

资源链接：[论文](https://arxiv.org/abs/2508.07999) |
[GitHub](https://github.com/ByteDance-Seed/WideSearch) |
[数据集](https://modelscope.cn/datasets/bytedance-community/WideSearch)

## 使用方法

默认本地 Bash 评估：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```

通过基准测试参数选择本地或 Docker 环境：

```python
TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    dataset_args={'wide_search': {'extra_params': {'environment_type': 'docker'}}},
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
)
```

使用 Bash 和 Fetch MCP 服务器复现论文的四次试验报告形式：

```python
import sys

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_MODEL',
    datasets=['wide_search'],
    repeats=4,
    agent_config=NativeAgentConfig(mcp_servers=[MCPServerConfigStdio(
        command=sys.executable,
        args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
        name='fetch',
    )]),
    judge_strategy='llm',
    judge_model_args={'model_id': 'YOUR_JUDGE_MODEL'},
))
```


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `wide_search` |
| **数据集ID** | [bytedance-community/WideSearch](https://modelscope.cn/datasets/bytedance-community/WideSearch/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2508.07999) |
| **标签** | `Agent`, `MultiTurn`, `Retrieval` |
| **指标** | `success_rate`, `row_precision`, `row_recall`, `row_f1`, `item_precision`, `item_recall`, `item_f1` |
| **默认示例数** | 0-shot |
| **评估划分** | `full` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 200 |
| 提示词长度（平均） | 631.01 字符 |
| 提示词长度（最小/最大） | 182 / 1807 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "cf605347",
      "content": "My son is about to start his university applications in 2025 for postgraduates but he’s still uncertain about both his major and which universities to apply to. Could you help me find the top five universities in each of the five broad subjec ... [TRUNCATED 691 chars] ... names in English. \nUse only Arabic numerals in the ranking, for example: 1.\nDon't ask me any questions, just output the results according to the columns without omitting cells arbitrarily. The output format is \n```markdown\n{data_content}\n```."
    }
  ],
  "target": "﻿Subject,University,Country,QS World University Rankings by Subject 2025,QS World University Rankings 2025,Times Higher Education  World University Rankings 2025,Home Page,Application Deadline,Application Fee\nArts & Humanities,Harvard Univers ... [TRUNCATED 2838 chars] ... nuary 5,$90\nSocial Sciences & Management,Massachusetts Institute of Technology,United States,4,1,2,https://www.mit.edu/,January 6 ,$75\nSocial Sciences & Management,University of Cambridge,United Kingdom,5,5,5,https://www.cam.ac.uk/,Oct 15,£60",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "bash",
      "description": "Execute a bash command inside the sandbox environment. Returns the combined stdout / stderr output of the command.",
      "parameters": {
        "properties": {
          "command": {
            "type": "string",
            "description": "The bash command to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "command"
        ]
      }
    }
  ],
  "metadata": {
    "instance_id": "ws_en_001",
    "language": "en",
    "evaluation": {
      "unique_columns": [
        "subject",
        "university"
      ],
      "required": [
        "subject",
        "university",
        "qsworlduniversityrankingsbysubject2025",
        "qsworlduniversityrankings2025",
        "timeshighereducationworlduniversityrankings2025",
        "homepage",
        "applicationdeadline",
        "applicationfee"
      ],
      "eval_pipeline": {
        "applicationdeadline": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nThe month and day must be correct"
        },
        "applicationfee": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nIf there are multiple fees in the reference answer, all must be included."
        },
        "homepage": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "url_match"
          ]
        },
        "subject": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "university": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankingsbysubject2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "timeshighereducationworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        }
      }
    }
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `max_steps` | `int` | `50` | 每个样本的最大函数调用 AgentLoop 步数。 |
| `environment_type` | `str` | `local` | 内置 bash 工具使用的环境。选项：['local', 'docker'] |
| `command_timeout` | `float` | `120.0` | 本地或 Docker 环境中 bash 命令的默认超时时间。 |
| `docker_image` | `str` | `python:3.11-slim` | 当 environment_type 为 docker 时使用的 Docker 镜像。 |
| `network_enabled` | `bool` | `True` | 是否允许 Docker 环境访问网络。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets wide_search \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":50}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['wide_search'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=50,
    ),
    dataset_args={
        'wide_search': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
