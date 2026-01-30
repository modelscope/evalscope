# SimpleQA


## 概述

SimpleQA 是由 OpenAI 设计的一个基准测试，用于评估语言模型准确回答简短事实性问题的能力。该基准专注于衡量事实准确性，并为正确、错误和未作答的答案提供了清晰的评分标准。

## 任务描述

- **任务类型**：事实型问答（Factual Question Answering）
- **输入**：简短的事实性问题
- **输出**：简洁的事实性答案
- **评分标准**：CORRECT（正确）、INCORRECT（错误）或 NOT_ATTEMPTED（未作答）

## 主要特点

- 问题简短、以事实为导向，且答案明确无歧义
- 提供清晰的评分标准用于准确性评估
- 能区分错误回答与主动放弃回答（abstention）
- 使用 LLM-as-judge 进行语义层面的答案比对
- 测试模型的事实知识掌握程度及其校准能力（calibration）

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 使用 LLM 作为裁判进行答案评分（基于语义匹配）
- 采用三分类判断：is_correct（正确）、is_incorrect（错误）、is_not_attempted（未作答）
- 若答案中包含正确信息，即使带有模糊表述（hedging）也可视为正确
- 测试模型在不确定时是否能恰当地承认不确定性

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `simple_qa` |
| **数据集ID** | [evalscope/SimpleQA](https://modelscope.cn/datasets/evalscope/SimpleQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `QA` |
| **指标** | `is_correct`, `is_incorrect`, `is_not_attempted` |
| **默认示例数量（Shots）** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,326 |
| 提示词长度（平均） | 118.47 字符 |
| 提示词长度（最小/最大） | 48 / 403 字符 |

## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "9dd57f4c",
      "content": "Answer the question:\n\nWho received the IEEE Frank Rosenblatt Award in 2010?"
    }
  ],
  "target": "Michio Sugeno",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "topic": "Science and technology",
    "answer_type": "Person",
    "urls": [
      "https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
      "https://ieeexplore.ieee.org/author/37271220500",
      "https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
      "https://www.nxtbook.com/nxtbooks/ieee/awards_2010/index.php?startid=21#/p/20"
    ]
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the question:

{question}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets simple_qa \
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
    datasets=['simple_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```