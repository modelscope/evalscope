# CommonsenseQA

## 概述

CommonsenseQA 是一个用于评估 AI 模型回答需要常识推理问题能力的基准测试。这些问题旨在考察模型是否具备问题中未明确说明的背景常识知识。

## 任务描述

- **任务类型**：多项选择常识推理
- **输入**：包含 5 个选项、需要常识知识的问题
- **输出**：正确答案的字母（A-E）
- **重点**：世界知识与常识推理

## 主要特点

- 问题基于 ConceptNet 知识图谱生成
- 需要多种类型的常识知识
- 每个问题提供 5 个选项
- 测试对日常概念及其关系的推理能力
- 所有问题均经过人工验证

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用简单的多项选择提示方式
- 在验证集（validation split）上进行评估
- 使用简单准确率（accuracy）作为评估指标

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `commonsense_qa` |
| **数据集 ID** | [extraordinarylab/commonsense-qa](https://modelscope.cn/datasets/extraordinarylab/commonsense-qa/summary) |
| **论文** | N/A |
| **标签** | `Commonsense`, `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,221 |
| 提示词长度（平均） | 326.11 字符 |
| 提示词长度（最小/最大） | 257 / 537 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "c9b96cfc",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E.\n\nA revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\n\nA) bank\nB) library\nC) department store\nD) mall\nE) new york"
    }
  ],
  "choices": [
    "bank",
    "library",
    "department store",
    "mall",
    "new york"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets commonsense_qa \
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
    datasets=['commonsense_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```