# SIQA

## 概述

SIQA（Social Interaction QA）是一个用于评估社会常识智能的基准测试，旨在衡量模型对人类行为及其社会影响的理解能力。与侧重于物理知识的基准不同，SIQA 专注于对人类行为的推理。

## 任务描述

- **任务类型**：社会常识推理
- **输入**：包含社交情境的上下文、一个问题以及三个选项
- **输出**：最符合社会规范的答案（A、B 或 C）
- **重点**：人类行为、动机及其社会影响

## 主要特点

- 测试社会智能与情感理解能力
- 问题围绕人类行为及其后果展开
- 涵盖动机、反应和社会规范
- 包含超过 33,000 个众包问答对
- 需要对人类心理进行推理

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用简单的多项选择提示方式
- 在验证集（validation split）上进行评估
- 使用简单准确率（accuracy）作为评估指标

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `siqa` |
| **数据集 ID** | [extraordinarylab/siqa](https://modelscope.cn/datasets/extraordinarylab/siqa/summary) |
| **论文** | N/A |
| **标签** | `Commonsense`, `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,954 |
| 提示词长度（平均） | 289.05 字符 |
| 提示词长度（最小/最大） | 242 / 509 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "8d09aab2",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C.\n\nWhat does Tracy need to do before this?\n\nA) make a new plan\nB) Go home and see Riley\nC) Find somewhere to go"
    }
  ],
  "choices": [
    "make a new plan",
    "Go home and see Riley",
    "Find somewhere to go"
  ],
  "target": "C",
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
    --datasets siqa \
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
    datasets=['siqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```