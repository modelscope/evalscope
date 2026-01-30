# AI2D


## 概述

AI2D（AI2 Diagrams）是一个用于评估人工智能系统理解与推理科学图表能力的基准数据集。该数据集包含来自科学教科书的5,000多个多样化图表，涵盖水循环、食物网和生物过程等主题。

## 任务描述

- **任务类型**：图表理解与视觉推理
- **输入**：科学图表图像 + 多选题
- **输出**：正确选项
- **领域**：科学教育、视觉推理、图表理解

## 主要特点

- 图表来源于真实的科学教科书
- 需要结合理解视觉布局、符号和文本标签
- 测试对图表元素之间关系的解读能力
- 采用包含具有挑战性干扰项的多选题格式
- 覆盖多样化的科学领域（生物学、物理学、地球科学）

## 评估说明

- 默认评估使用 **test** 划分
- 主要指标：多选题的 **准确率（Accuracy）**
- 使用思维链（Chain-of-Thought, CoT）提示进行推理
- 需同时理解文本标签和视觉元素


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ai2d` |
| **数据集ID** | [lmms-lab/ai2d](https://modelscope.cn/datasets/lmms-lab/ai2d/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,088 |
| 提示词长度（平均） | 324.64 字符 |
| 提示词长度（最小/最大） | 256 / 1024 字符 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 3,088 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 177x131 - 1500x1500 |
| 格式 | png |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "789e28fa",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nwhich of these define dairy item\n\nA) c\nB) D\nC) b\nD) a"
        },
        {
          "image": "[BASE64_IMAGE: png, ~226.2KB]"
        }
      ]
    }
  ],
  "choices": [
    "c",
    "D",
    "b",
    "a"
  ],
  "target": "B",
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

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ai2d \
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
    datasets=['ai2d'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```