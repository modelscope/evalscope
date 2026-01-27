# GSM8K

GSM8K（Grade School Math 8K）是一个由人工编写的问题集，包含 8.5K 个高质量、语言表达多样化的中小学数学应用题。

**任务类型**：数学推理  
**难度级别**：中小学水平  
**答案格式**：数值答案，用 \boxed{{}} 包裹

该数据集旨在支持对需要多步推理的基础数学问题进行问答。每个问题包含一个自然语言提问和一个数值答案。这些问题考察基本的算术运算能力，通常需要 2 到 8 个步骤才能解答。

**主要特点**：
- 高质量的人工编写题目
- 语言表达形式多样
- 需要多步推理
- 答案明确且为数值型

## 概述

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gsm8k` |
| **数据集ID** | [AI-ModelScope/gsm8k](https://modelscope.cn/datasets/AI-ModelScope/gsm8k/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2110.14168) |
| **标签** | `Math`, `Reasoning` |
| **评估指标** | `acc` |
| **默认示例数量（Shots）** | 4-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,319 |
| 提示词长度（平均） | 1966.87 字符 |
| 提示词长度（最小/最大） | 1800 / 2575 字符 |

## 子集

- `main`（1,319 个样本）

## 样例示例

**子集**: `main`

```json
{
  "input": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market."
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
请逐步推理，并将最终答案放在 \boxed{{}} 中。
```

## 使用方法

```python
from evalscope import run_task

results = run_task(
    model='your-model',
    datasets=['gsm8k'],
)
```