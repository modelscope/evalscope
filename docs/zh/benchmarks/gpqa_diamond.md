# GPQA-Diamond


## 概述

GPQA（Graduate-Level Google-Proof Q&A）Diamond 是一个极具挑战性的基准测试，包含 198 道由生物学、物理学和化学领域的专家编写的多项选择题。这些问题设计得极为困难，需要博士级别的专业知识才能正确作答。

## 任务描述

- **任务类型**：专家级多项选择问答
- **输入**：研究生水平的科学问题，附带 4 个选项
- **输出**：单个正确答案字母（A、B、C 或 D）
- **领域**：生物学、物理学、化学

## 主要特点

- 198 道题目均由相关领域的博士专家编写并验证
- 题目“无法通过 Google 轻易查到”——难以通过简单搜索获得答案
- 旨在测试深层次的领域知识与推理能力
- Diamond 子集代表了最高质量的问题
- 人类专家平均准确率约为 65%，非专家约为 34%

## 评估说明

- 默认配置使用 **0-shot** 或 **5-shot** 评估
- 支持思维链（Chain-of-Thought, CoT）提示以提升推理能力
- 评估过程中答案选项会随机打乱
- 仅使用训练集（验证集为私有）
- 是衡量专家级推理能力的高难度基准测试

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gpqa_diamond` |
| **数据集 ID** | [AI-ModelScope/gpqa_diamond](https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 198 |
| 提示词长度（平均） | 841.15 字符 |
| 提示词长度（最小/最大） | 340 / 5845 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "82b448a9",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nTwo quantum states wi ... [TRUNCATED] ...  and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\n\nA) 10^-4 eV\nB) 10^-9 eV\nC) 10^-8 eV\nD) 10^-11 eV"
    }
  ],
  "choices": [
    "10^-4 eV",
    "10^-9 eV",
    "10^-8 eV",
    "10^-11 eV"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "",
  "metadata": {
    "correct_answer": "10^-4 eV",
    "incorrect_answers": [
      "10^-11 eV",
      "10^-8 eV\n",
      "10^-9 eV"
    ]
  }
}
```

*注：部分内容因展示需要已被截断。*

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
    --datasets gpqa_diamond \
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
    datasets=['gpqa_diamond'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```