# AI2D

## 概述

AI2D（AI2 图表）是一个用于评估AI系统理解和推理科学图表能力的基准数据集。它包含来自科学教科书中的5,000多张多样化的图表，涵盖水循环、食物网和生物过程等主题。

## 任务描述

- **任务类型**：图表理解和视觉推理
- **输入**：科学图表图像 + 多项选择题
- **输出**：正确答案选项
- **领域**：科学教育、视觉推理、图表理解

## 主要特点

- 图表来源于真实的科学教科书
- 需要联合理解视觉布局、符号和文本标签
- 测试对图表元素之间关系的解释
- 多项选择格式，具有挑战性的干扰项
- 涵盖多种科学领域（生物学、物理学、地球科学）

## 评估说明

- 默认评估使用 **test** 分割
- 主要指标：多项选择题的 **准确率**
- 使用Chain-of-Thought (CoT) 提示进行推理
- 需要理解文本标签和视觉元素

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ai2d` |
| **数据集ID** | [evalscope/ai2d](https://modelscope.cn/datasets/evalscope/ai2d/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `accuracy` |
| **默认提示次数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,088 |
| 提示词长度（平均值） | 324.64 字符 |
| 提示词长度（最小/最大值） | 256 / 1024 字符 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 3,088 |
| 每个样本的图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 177x131 - 1500x1500 |
| 格式 | jpeg |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "ce4b20b6",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhat is between the head and abdomen?\n\nA) Antenna\nB) Simple eye\nC) Spiracle\nD) Thorax"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~59.4KB]"
        }
      ]
    }
  ],
  "choices": [
    "Antenna",
    "Simple eye",
    "Spiracle",
    "Thorax"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ai2d \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['ai2d'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
