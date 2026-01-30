# MathVision

## 概述

MATH-Vision（MATH-V）是一个精心整理的数据集，包含3,040道高质量的数学问题，这些问题均来自真实的数学竞赛，并配有视觉上下文（如图表、图形等），用于评估多模态环境下的数学推理能力。

## 任务描述

- **任务类型**：视觉数学推理
- **输入**：带有视觉上下文（图示、图形、图表）的数学问题
- **输出**：数值答案或选项字母
- **领域**：包含视觉元素的竞赛数学

## 主要特点

- 问题源自真实的数学竞赛
- 包含3,040道高质量、带视觉上下文的问题
- 划分为五个难度等级（1-5），便于细粒度分析
- 支持多项选择题和自由作答两种形式
- 提供详细的参考解答

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 按难度划分子集：`level 1` 至 `level 5`
- 主要评估指标：**准确率（Accuracy）**，采用数值比较方式
- 自由作答题答案需使用 \boxed{} 格式（不含单位）
- 多项选择题采用思维链（CoT）提示，答案为选项字母

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `math_vision` |
| **数据集ID** | [evalscope/MathVision](https://modelscope.cn/datasets/evalscope/MathVision/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,040 |
| 提示词长度（平均） | 423.91 字符 |
| 提示词长度（最小/最大） | 93 / 1581 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `level 1` | 237 | 357.8 | 133 | 854 |
| `level 2` | 738 | 397.17 | 127 | 885 |
| `level 3` | 551 | 433.46 | 93 | 1402 |
| `level 4` | 830 | 434.51 | 134 | 946 |
| `level 5` | 684 | 455.12 | 129 | 1581 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 3,040 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 54x40 - 2520x8526 |
| 图像格式 | jpeg |

## 样例示例

**子集**: `level 1`

```json
{
  "input": [
    {
      "id": "606d1e5a",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E. Think step by step before answering.\n\nWhich kite has the longest string?\n<image1>\n\nA) A\nB) B\nC) C\nD) D\nE) E"
        },
        {
          "image": "[BASE64_IMAGE: jpg, ~38.6KB]"
        }
      ]
    }
  ],
  "choices": [
    "A",
    "B",
    "C",
    "D",
    "E"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "subset_key": "level 1",
  "metadata": {
    "id": "3",
    "image": "images/3.jpg",
    "solution": null,
    "level": 1,
    "question_type": "multi_choice",
    "subject": "metric geometry - length"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}} without units.
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets math_vision \
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
    datasets=['math_vision'],
    dataset_args={
        'math_vision': {
            # subset_list: ['level 1', 'level 2', 'level 3']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```