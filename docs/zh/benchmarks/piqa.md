# PIQA

## 概述

PIQA（Physical Interaction QA）是一个用于评估 AI 模型对物理常识理解能力的基准测试，重点考察模型对物理世界中物体如何相互作用以及在人为操作下会发生什么的理解。

## 任务描述

- **任务类型**：物理常识推理
- **输入**：一个目标/问题及两个可能的解决方案
- **输出**：更符合物理常识的解决方案（A 或 B）
- **关注点**：物理世界知识与直觉物理推理

## 主要特点

- 测试对物理对象属性的理解
- 在合理与不合理方案之间进行二元选择
- 需要直觉物理推理能力
- 覆盖日常物理场景
- 经过对抗性过滤以减少数据偏差

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用简单的多项选择提示方式
- 在验证集（validation split）上进行评估
- 使用简单准确率（accuracy）作为评估指标

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `piqa` |
| **数据集ID** | [extraordinarylab/piqa](https://modelscope.cn/datasets/extraordinarylab/piqa/summary) |
| **论文** | N/A |
| **标签** | `Commonsense`, `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估集** | `validation` |
| **训练集** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,838 |
| 提示词长度（平均） | 426.52 字符 |
| 提示词长度（最小/最大） | 220 / 2335 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a0600392",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nHow do I ready a guinea pig cage for it's new occupants?\n ... [TRUNCATED] ... ps, you will also need to supply it with a water bottle and a food dish.\nB) Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
    }
  ],
  "choices": [
    "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.",
    "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

*注：部分内容为显示目的已截断。*

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
    --datasets piqa \
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
    datasets=['piqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```