# MMLU-Pro

## 概述

MMLU-Pro 是 MMLU 的增强版本，具有更高的难度和更强的推理要求。它将每道题的答案选项从 4 个增加到 10 个，并包含更具挑战性的问题，需要在 14 个不同领域中进行更深入的推理。

## 任务描述

- **任务类型**：多项选择题问答（10 个选项）
- **输入**：包含最多 10 个答案选项的问题
- **输出**：单个正确答案字母（A-J）
- **领域**：涵盖科学、法律、工程、心理学等 14 个学科

## 主要特点

- 每道题提供 10 个答案选项（原始 MMLU 为 4 个）
- 问题更具挑战性，需要更强的推理能力
- 包含思维链（Chain-of-Thought, CoT）解释
- 覆盖 14 个多样化主题
- 随机猜测的正确率显著降低（10% 对比 25%）

## 评估说明

- 默认配置使用 **5-shot** 示例
- 答案应遵循 "ANSWER: [LETTER]" 格式
- 使用特定学科的 few-shot 示例
- 鼓励逐步推理
- 在测试集上评估，并使用验证集提供 few-shot 示例

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmlu_pro` |
| **数据集ID** | [TIGER-Lab/MMLU-Pro](https://modelscope.cn/datasets/TIGER-Lab/MMLU-Pro/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 12,032 |
| 提示词长度（平均） | 4959.66 字符 |
| 提示词长度（最小/最大） | 3048 / 12100 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `computer science` | 410 | 5184.05 | 4696 | 7303 |
| `math` | 1,351 | 4853.54 | 4574 | 7519 |
| `chemistry` | 1,132 | 4567.2 | 4182 | 5975 |
| `engineering` | 969 | 3595.66 | 3048 | 5775 |
| `law` | 1,101 | 7001.35 | 5581 | 9200 |
| `biology` | 717 | 5881.08 | 5179 | 7850 |
| `health` | 818 | 4624.44 | 4134 | 9310 |
| `physics` | 1,299 | 4389.59 | 4004 | 6511 |
| `business` | 789 | 4646.01 | 4316 | 6915 |
| `philosophy` | 499 | 3760.1 | 3360 | 5289 |
| `economics` | 844 | 4680.21 | 4118 | 6907 |
| `other` | 924 | 3730.86 | 3316 | 6108 |
| `psychology` | 798 | 5274.19 | 4656 | 6942 |
| `history` | 381 | 9919.94 | 8711 | 12100 |

## 样例示例

**子集**: `computer science`

```json
{
  "input": [
    {
      "id": "79336f87",
      "content": "The following are multiple choice questions (with answers) about computer science. Think step by step and then finish your answer with 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the correct letter choice.\n\nQuestion:\nA certain pipel ... [TRUNCATED] ...  random index of a larger value.\nG) The method should be written to return the largest index among larger values.\nH) The method should be written on the assumption that there is only one value in the array that is larger than the given item.\n"
    }
  ],
  "choices": [
    "The method should return an error if more than one larger value is found.",
    "The specification should be modified to indicate what should be done if there is more than one index of larger values.",
    "The method should be written to output a message if more than one larger value is found.",
    "The method should be written so as to return the index of every occurrence of a larger value.",
    "The method should be written to return the last occurrence of a larger value.",
    "The method should return a random index of a larger value.",
    "The method should be written to return the largest index among larger values.",
    "The method should be written on the assumption that there is only one value in the array that is larger than the given item."
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "computer science",
  "metadata": {
    "cot_content": "",
    "subject": "computer science",
    "question_id": 10356
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}

```

<details>
<summary>Few-shot 模板</summary>

```text
The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the correct letter choice.

{examples}
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}

```

</details>

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmlu_pro \
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
    datasets=['mmlu_pro'],
    dataset_args={
        'mmlu_pro': {
            # subset_list: ['computer science', 'math', 'chemistry']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```