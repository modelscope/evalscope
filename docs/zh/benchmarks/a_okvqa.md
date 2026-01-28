# A-OKVQA


## 概述

A-OKVQA（Augmented OK-VQA）是一个用于评估视觉问答（VQA）中常识推理与外部世界知识能力的基准测试。它超越了仅依赖图像内容的基础VQA任务，要求模型利用广泛的世界常识和事实性知识进行推理。

## 任务描述

- **任务类型**：基于知识推理的视觉问答
- **输入**：图像 + 需要外部知识的自然语言问题
- **输出**：答案（多项选择或开放式）
- **领域**：常识推理、事实性知识、视觉理解

## 主要特点

- 要求模型进行超出直接视觉观察的常识推理
- 结合视觉理解与外部世界知识
- 包含多项选择题和开放式问题两种格式
- 问题附带解释推理过程的理由（rationales）
- 相比标准VQA基准更具挑战性

## 评估说明

- 默认使用 **验证集（validation）** 进行评估
- 主要指标：多项选择题的 **准确率（Accuracy）**
- 使用思维链（Chain-of-Thought, CoT）提示以提升推理能力
- 问题需要基于图像之外的知识进行推理

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `a_okvqa` |
| **数据集ID** | [HuggingFaceM4/A-OKVQA](https://modelscope.cn/datasets/HuggingFaceM4/A-OKVQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,145 |
| 提示词长度（平均） | 310.84 字符 |
| 提示词长度（最小/最大） | 276 / 405 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,145 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 305x229 - 640x640 |
| 格式 | jpeg |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "3d2b0351",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhat is in the motorcyclist's mouth?\n\nA) toothpick\nB) food\nC) popsicle stick\nD) cigarette"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~53.4KB]"
        }
      ]
    }
  ],
  "choices": [
    "toothpick",
    "food",
    "popsicle stick",
    "cigarette"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "22jbM6gDxdaMaunuzgrsBB",
    "direct_answers": "['cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette']",
    "difficult_direct_answer": false,
    "rationales": [
      "He's smoking while riding.",
      "The motorcyclist has a lit cigarette in his mouth while he rides on the street.",
      "The man is smoking."
    ]
  }
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

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets a_okvqa \
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
    datasets=['a_okvqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```