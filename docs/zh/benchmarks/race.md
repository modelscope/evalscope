# RACE


## 概述

RACE（ReAding Comprehension from Examinations）是一个大规模阅读理解基准数据集，收集自中国初中和高中英语考试题目。该数据集用于测试综合阅读理解能力。

## 任务描述

- **任务类型**：阅读理解（多项选择题）
- **输入**：文章段落、问题及4个选项
- **输出**：正确答案字母（A、B、C 或 D）
- **难度级别**：初中和高中

## 主要特点

- 包含28,000+篇文章和100,000道问题
- 使用真实考试题目，确保难度的真实性
- 包含两个子集：`middle`（较简单）和 `high`（较难）
- 考察多种阅读理解技能（推理、词汇、主旨等）
- 文章主题和题型丰富多样

## 评估说明

- 默认配置使用 **3-shot** 示例
- 最大 few-shot 数量为3（受上下文长度限制）
- 使用思维链（Chain-of-Thought, CoT）提示方法
- 提供两个子集：`high` 和 `middle`
- 在测试集（test split）上进行评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `race` |
| **数据集ID** | [evalscope/race](https://modelscope.cn/datasets/evalscope/race/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 3-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,934 |
| 提示词长度（平均） | 7217.34 字符 |
| 提示词长度（最小/最大） | 3685 / 11131 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `high` | 3,498 | 8279.47 | 6585 | 11131 |
| `middle` | 1,436 | 4630.07 | 3685 | 6032 |

## 样例示例

**子集**: `high`

```json
{
  "input": [
    {
      "id": "706382e9",
      "content": "Here are some examples of how to answer similar questions:\n\nArticle:\nLast week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects  they thought they had.\nGiven that I teach stud ... [TRUNCATED] ... I owe my life to her,\" said Nancy with tears.\nQuestion:\nWhat did Nancy try to do before she fell over?\n\nA) Measure the depth of the river\nB) Look for a fallen tree trunk\nC) Protect her cows from being drowned\nD) Run away from the flooded farm"
    }
  ],
  "choices": [
    "Measure the depth of the river",
    "Look for a fallen tree trunk",
    "Protect her cows from being drowned",
    "Run away from the flooded farm"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "example_id": "high19432.txt"
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

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets race \
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
    datasets=['race'],
    dataset_args={
        'race': {
            # subset_list: ['high', 'middle']  # 可选，用于指定评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```