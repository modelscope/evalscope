# OfficeQA


## 概述

OfficeQA 是由 Databricks 构建的一个基于真实文档的推理基准测试，用于评估模型/智能体在 1939–2025 年美国财政部公告（U.S. Treasury Bulletin）文档上执行端到端 grounded reasoning 任务的性能。

## 任务描述

- **任务类型**：基于智能体的文档问答（通过 grep/search 在语料库中检索）
- **输入**：一个问题 + 通过 bash 工具访问已解析的财政部公告文本文件
- **输出**：精确答案（数值、文本或结构化数据）
- **评估模式**：智能体使用 bash 工具（如 grep、cat 等）在语料库上进行检索

## 主要特性

- 包含两个子集：`officeqa_pro`（133 个问题，难度高，默认使用）和 `officeqa_full`（246 个问题，包含简单与困难问题）
- 语料库：约 900 个已解析的财政部公告文本文件（总计约 460MB）
- 智能体使用 bash 工具（grep、cat、head 等）搜索语料库
- 评分采用模糊数值匹配，支持可配置容差（默认为 1%）

## 评估说明

- 智能体可访问语料库目录中的已解析 .txt 文件
- 每个问题的 `source_files` 字段指明了包含答案的文档
- 使用从官方 reward.py 改编的**基于规则的评分机制**
- 数值答案允许 1% 的相对误差容差
- 文本答案采用不区分大小写的子串匹配

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `officeqa` |
| **数据集ID** | [evalscope/officeqa](https://modelscope.cn/datasets/evalscope/officeqa/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `Knowledge`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 133 |
| 提示词长度（平均） | 443.06 字符 |
| 提示词长度（最小/最大） | 165 / 1186 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a6357de6",
      "content": "What were the total expenditures (in millions of nominal dollars) for U.S national defense in the calendar year of 1940?\nPlease provide a precise and concise answer."
    }
  ],
  "target": "2,602",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "uid": "UID0001",
    "source_files": "treasury_bulletin_1941_01.txt",
    "difficulty": "hard"
  }
}
```

## 提示模板

**提示模板:**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets officeqa \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":15}' \
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
    datasets=['officeqa'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=15,
    ),
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
