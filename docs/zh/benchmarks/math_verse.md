# MathVerse

## 概述

MathVerse 是一个全面的视觉数学基准测试，旨在对多模态大语言模型（MLLMs）进行公平且深入的评估。它包含 2,612 道高质量、多学科的带图数学题，并根据不同信息模态转换为 15K 个测试样本。

## 任务描述

- **任务类型**：视觉数学推理
- **输入**：带图的数学问题 + 问题（多选题或自由作答）
- **输出**：答案（多选题为选项字母，自由作答为数值或表达式）
- **领域**：带视觉图表的多学科数学

## 核心特性

- 2,612 道题目，每道题生成 6 个版本（共 15K 个样本）
- 测试 MLLM 是否真正理解用于数学推理的视觉图表
- 题目版本根据对视觉信息的依赖程度划分：
  - **Text Dominant（文本主导）**：大部分信息在文本中
  - **Text Lite（文本轻量）**：文本与视觉信息均衡
  - **Vision Intensive（视觉密集）**：更依赖视觉信息
  - **Vision Dominant（视觉主导）**：主要依赖视觉信息
  - **Vision Only（纯视觉）**：所有信息均在图表中
- 支持多选题和自由作答两种形式

## 评估说明

- 默认评估使用 **testmini** 划分
- 主要指标：**准确率（Accuracy）**，采用数值比较
- 自由作答答案使用 \boxed{} 格式
- 使用 LLM 作为裁判进行答案验证
- 按题目版本分别报告结果，便于深入分析

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `math_verse` |
| **数据集ID** | [evalscope/MathVerse](https://modelscope.cn/datasets/evalscope/MathVerse/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认提示数量** | 0-shot |
| **评估划分** | `testmini` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,940 |
| 提示词长度（平均） | 274.2 字符 |
| 提示词长度（最小/最大） | 70 / 1535 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Text Dominant` | 788 | 369.63 | 122 | 1535 |
| `Text Lite` | 788 | 294.77 | 78 | 1397 |
| `Vision Intensive` | 788 | 280.39 | 78 | 1350 |
| `Vision Dominant` | 788 | 272.11 | 78 | 1356 |
| `Vision Only` | 788 | 154.1 | 70 | 222 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 3,940 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 63x70 - 6840x3549 |
| 格式 | jpeg, png |

## 样例示例

**子集**: `Text Dominant`

```json
{
  "input": [
    {
      "id": "a3189330",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.\n\nAs shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°"
        },
        {
          "image": "[BASE64_IMAGE: png, ~1.6KB]"
        }
      ]
    }
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "Text Dominant",
  "metadata": {
    "sample_index": "1",
    "problem_index": "1",
    "problem_version": "Text Dominant",
    "question_type": "multi-choice",
    "query_wo": "Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.\nQuestion: As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°",
    "query_cot": "Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°",
    "question_for_eval": "As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets math_verse \
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
    datasets=['math_verse'],
    dataset_args={
        'math_verse': {
            # subset_list: ['Text Dominant', 'Text Lite', 'Vision Intensive']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```