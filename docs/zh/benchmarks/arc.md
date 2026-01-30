# ARC


## 概述

ARC（AI2 Reasoning Challenge）是一个用于评估 AI 模型科学问答能力的基准测试。它包含来自 3 至 9 年级的多项选择题科学问题，并根据难度分为 Easy（简单）集和 Challenge（挑战）集。

## 任务描述

- **任务类型**：多项选择科学问答
- **输入**：一道包含 3-5 个选项的科学问题
- **输出**：正确答案的字母（A、B、C、D 或 E）
- **难度级别**：ARC-Easy 和 ARC-Challenge

## 主要特点

- 包含 7,787 道来自标准化考试的科学问题（年级 3-9）
- ARC-Easy：可通过检索或词语共现回答的问题
- ARC-Challenge：需要更深层次推理的问题
- 问题涵盖物理、化学、生物和地球科学
- 旨在测试事实性知识与推理能力

## 评估说明

- 默认配置使用 **0-shot** 评估
- 提供两个子集：`ARC-Easy` 和 `ARC-Challenge`
- Challenge 子集常用于排行榜比较
- 支持使用训练集样例进行 few-shot 评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `arc` |
| **数据集 ID** | [allenai/ai2_arc](https://modelscope.cn/datasets/allenai/ai2_arc/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认 Shots** | 0-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,548 |
| 提示词长度（平均） | 424.43 字符 |
| 提示词长度（最小/最大） | 253 / 1157 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `ARC-Easy` | 2,376 | 409.9 | 253 | 1157 |
| `ARC-Challenge` | 1,172 | 453.91 | 253 | 1111 |

## 样例示例

**子集**: `ARC-Easy`

```json
{
  "input": [
    {
      "id": "76edd7a5",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich statement best explains why photosynthesis is the foundation of most food webs?\n\nA) Sunlight is the source of energy for nearly all ecosystems.\nB) Most ecosystems are found on land instead of in water.\nC) Carbon dioxide is more available than other gases.\nD) The producers in all ecosystems are plants."
    }
  ],
  "choices": [
    "Sunlight is the source of energy for nearly all ecosystems.",
    "Most ecosystems are found on land instead of in water.",
    "Carbon dioxide is more available than other gases.",
    "The producers in all ecosystems are plants."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "Mercury_417466"
  }
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
    --datasets arc \
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
    datasets=['arc'],
    dataset_args={
        'arc': {
            # subset_list: ['ARC-Easy', 'ARC-Challenge']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```