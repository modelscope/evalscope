# OfficeQA


## 概述

OfficeQA 是一个具有挑战性的问答基准测试，旨在评估大语言模型（LLMs）基于官方文件（例如美国财政部公告）回答问题的能力。该基准测试考察模型从结构化的金融和政府数据中精确提取信息并进行推理的能力。

## 任务描述

- **任务类型**：基于文档的问答（Document-based Question Answering）
- **输入**：关于官方/政府文件的问题
- **输出**：精确答案（数值、文本或结构化数据）
- **难度**：所有问题均被评定为“困难”

## 主要特点

- 包含 133 个源自真实美国财政部公告和政府文件的问题
- 答案包括带格式的数字（例如 "2,602"）、列表和文本
- 考察从官方文档中精确提取事实信息的能力
- 所有问题均需对结构化数据进行仔细解读

## 评估说明

- 由于答案格式多样，采用 **LLM judge** 进行评分
- 答案可能是数字、格式化数值或列表
- 采用零样本（zero-shot）评估（无少样本示例）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `officeqa` |
| **数据集ID** | [evalscope/officeqa](https://modelscope.cn/datasets/evalscope/officeqa/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `QA` |
| **指标** | `acc` |
| **默认样本数** | 0-shot |
| **评估分割** | `train` |


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
Please provide a precise and concise answer.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets officeqa \
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
    datasets=['officeqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```