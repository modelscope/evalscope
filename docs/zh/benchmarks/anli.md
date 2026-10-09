# ANLI


## 概述

在三个独立收集的轮次上进行对抗性自然语言推理。

## 任务描述

- **任务类型**：单选分类
- **输入**：前提（Premise）和假设（Hypothesis）
- **输出**：蕴含（Entailment）、中立（Neutral）或矛盾（Contradiction）
- **领域**：英语语言理解

## 主要特性

- 公开数据集：Hugging Face 上的 `facebook/anli`
- 保留原始标签和完整的任务上下文
- 支持聊天生成模型和文本 System One Choice 模型

## 评估说明

- 分别报告每轮的准确率以及样本加权的总体得分。测试解释不会作为模型输入。
- 默认为 0-shot；若存在训练集，可配置训练示例
- 系统提示将转换为选择题任务指令，不使用原生的聊天角色层级结构
- 评估语义版本：v1.0

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `anli` |
| **数据集ID** | [facebook/anli](https://huggingface.co/datasets/facebook/anli) |
| **论文** | N/A |
| **标签** | `MCQ` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,200 |
| 提示词长度（平均） | 785.53 字符 |
| 提示词长度（最小/最大） | 541 / 1303 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `r1` | 1,000 | 791.1 | 684 | 987 |
| `r2` | 1,000 | 790.79 | 681 | 1015 |
| `r3` | 1,200 | 776.5 | 541 | 1303 |

## 样例示例

**子集**: `r1`

```json
{
  "input": [
    {
      "id": "07cbb889",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C.\n\nPremise:\nErnest Jones is a British jeweller and watchma ... [TRUNCATED 261 chars] ... ones store was opened on the continent of Europe.\n\nA) Entailment: the hypothesis follows from the premise.\nB) Neutral: the premise does not determine whether the hypothesis is true.\nC) Contradiction: the hypothesis conflicts with the premise."
    }
  ],
  "choices": [
    "Entailment: the hypothesis follows from the premise.",
    "Neutral: the premise does not determine whether the hypothesis is true.",
    "Contradiction: the hypothesis conflicts with the premise."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "uid": "4aae63a8-fcf7-406c-a2f3-50c31c5934a9",
    "round": "r1"
  }
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
    --datasets anli \
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
    datasets=['anli'],
    dataset_args={
        'anli': {
            # subset_list: ['r1', 'r2', 'r3']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
