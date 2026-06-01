# GAIA


## 概述

GAIA（General AI Assistants）是一个包含 450 多个问题的基准测试，旨在评估具备工具使用、网页浏览和多步推理能力的新一代大语言模型（LLM）。每个问题都有一个明确的简短答案，并被划分为三个难度等级之一。

## 任务描述

- **任务类型**：工具使用型智能体（多轮交互）
- **输入**：自然语言问题，可选附带一个引用附件文件（PDF / xlsx / 图像 / 音频 / ...）
- **输出**：一个简短的最终答案（数字 / 短语 / 逗号分隔的列表）
- **数据划分**：``validation``（答案公开）和 ``test``（答案私有 — 跳过评估）

## 核心特性

- 基于 ReAct 智能体循环，内置单个 ``bash`` 工具，运行于 Docker 沙箱中（默认镜像为 ``python:3.11``，已预装 ``curl`` / ``wget`` / ``git``）。
- 附件文件以只读方式挂载至沙箱内的 ``/shared_files`` 目录。
- 评分器直接移植自官方 GAIA 排行榜的规则逻辑（不使用 LLM 作为评判）。
- 默认从 ModelScope 下载数据集（``gaia-benchmark/GAIA``）；设置 ``dataset_hub='huggingface'`` 可改为从 Hugging Face 加载。

## 评估说明

- 需要在本地运行 Docker 守护进程（或通过 ms_enclave 配置使用远程沙箱引擎）。
- ``extra_params.max_steps`` 限制智能体循环的最大步数（默认为 50）。
- ``extra_params.command_timeout`` 设置每条 ``bash`` 命令的超时时间（默认 180 秒，与 inspect_ai 一致）。
- 默认启用网络访问 — 许多问题需要进行网页浏览或搜索。
- 使用 ``subset_list`` 可限定特定难度级别，例如 ``['2023_level1']``、``['2023_level1', '2023_level2']`` 或默认的 ``['2023_all']``。
- [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/gaia.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gaia` |
| **数据集ID** | [gaia-benchmark/GAIA](https://modelscope.cn/datasets/gaia-benchmark/GAIA/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `MultiTurn`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 165 |
| 提示词长度（平均） | 861.35 字符 |
| 提示词长度（最小/最大） | 596 / 2582 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `2023_level1` | 53 | 906.53 | 604 | 2582 |
| `2023_level2` | 86 | 816.66 | 596 | 1275 |
| `2023_level3` | 26 | 917.08 | 621 | 1497 |

## 样例示例

**子集**: `2023_level1`

```json
{
  "input": [
    {
      "id": "93e8257c",
      "content": "Please answer the question below. You should:\n\n- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.\n- If the answer is a number, return only ... [TRUNCATED 431 chars] ... Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."
    }
  ],
  "target": "17",
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
    "task_id": "e1fc63a2-da7a-432f-be78-7c4a95598703",
    "level": "1",
    "file_name": "",
    "file_path": "",
    "Annotator Metadata": {
      "Steps": "1. Googled Eliud Kipchoge marathon pace to find 4min 37sec/mile\n2. Converted into fractions of hours.\n3. Found moon periapsis in miles (225,623 miles).\n4. Multiplied the two to find the number of hours and rounded to the nearest 100 hours.",
      "Number of steps": "4",
      "How long did this take?": "20 Minutes",
      "Tools": "1. A web browser.\n2. A search engine.\n3. A calculator.",
      "Number of tools": "3"
    }
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `max_steps` | `int` | `50` | 每个样本允许的最大智能体步数。 |
| `command_timeout` | `float` | `180.0` | 每条 bash 命令的默认超时时间（秒）。 |
| `docker_image` | `str` | `python:3.11` | 用作每个样本沙箱环境的 Docker 镜像。 |
| `network_enabled` | `bool` | `True` | 是否允许沙箱访问网络（GAIA 中涉及浏览的问题需要此选项）。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gaia \
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
    datasets=['gaia'],
    dataset_args={
        'gaia': {
            # subset_list: ['2023_level1', '2023_level2', '2023_level3']  # 可选，用于评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```