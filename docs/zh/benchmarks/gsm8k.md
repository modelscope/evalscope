# GSM8K

GSM8K（Grade School Math 8K）是一个包含 8.5K 道高质量、语言多样化的中小学数学应用题的数据集，由人工编写而成。

该数据集旨在支持对需要多步推理的基础数学问题进行问答任务。每个问题包含一个自然语言提问和一个数值答案。这些问题考察基本的算术运算，通常需要 2-8 个步骤才能解答。

**主要特点**：
- 高质量的人工编写题目
- 语言表达形式多样
- 需要多步推理
- 答案为明确的数值

## 概述

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gsm8k` |
| **数据集ID** | [AI-ModelScope/gsm8k](https://modelscope.cn/datasets/AI-ModelScope/gsm8k/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2110.14168) |
| **标签** | `Math`, `Reasoning` |
| **评估指标** | `acc` |
| **默认示例数量** | 4-shot |
| **评估集** | `test` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,319 |
| 提示词长度（平均） | 1966.87 字符 |
| 提示词长度（最小/最大） | 1800 / 2575 字符 |

## 样例示例

**子集**: `main`

```json
{
  "input": [
    {
      "id": "3d025da2",
      "content": "Here are some examples of how to solve similar problems:\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nReasoning:\nNatalia sold 48/ ... [TRUNCATED] ... ds every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market."
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

<details>
<summary>少样本（Few-shot）模板</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}

{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

</details>

## 使用方法

```python
from evalscope import run_task

results = run_task(
    model='your-model',
    datasets=['gsm8k'],
)
```