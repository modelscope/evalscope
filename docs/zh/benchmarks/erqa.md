# ERQA


## 概述

ERQA（Embodied Reasoning QA）是一个用于评估多模态大语言模型空间推理与具身理解能力的基准测试。该基准测试考察模型在以自我为中心（egocentric）的机器人场景中，对轨迹、动作、空间关系以及任务规划进行推理的能力。

## 任务描述

- **任务类型**：具身空间推理（多项选择题）
- **输入**：以自我为中心的图像（单张或多张）+ 多项选择题（A/B/C/D）
- **输出**：单个答案字母（A/B/C/D）
- **领域**：机器人学、空间推理、具身人工智能

## 主要特点

- 包含8个推理类别的400道题目
- 支持多图像输入（部分题目需跨多个视角进行推理）
- 类别包括：轨迹推理（Trajectory Reasoning）、动作推理（Action Reasoning）、指向（Pointing）、状态估计（State Estimation）、空间推理（Spatial Reasoning）、多视角推理（Multi-view Reasoning）、任务推理（Task Reasoning）及其他（Other）
- 图像视角来自机器人操作场景的自我中心视角

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 在 **test** 划分上进行评估
- 主要指标：多项选择题的 **准确率（Accuracy）**
- 答案为单个字母（A/B/C/D）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `erqa` |
| **数据集ID** | [evalscope/ERQA](https://modelscope.cn/datasets/evalscope/ERQA/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认Shots数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 400 |
| 提示词长度（平均） | 289.05 字符 |
| 提示词长度（最小/最大） | 147 / 834 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Trajectory Reasoning` | 66 | 307.3 | 161 | 834 |
| `Action Reasoning` | 72 | 271.97 | 175 | 503 |
| `Pointing` | 34 | 300.82 | 178 | 676 |
| `State Estimation` | 55 | 314.84 | 147 | 735 |
| `Spatial Reasoning` | 84 | 286.21 | 152 | 819 |
| `Multi-view Reasoning` | 37 | 277.78 | 196 | 475 |
| `Task Reasoning` | 38 | 280.18 | 152 | 737 |
| `Other` | 14 | 231.79 | 188 | 409 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 630 |
| 每样本图像数 | 最小: 1, 最大: 16, 平均: 1.57 |
| 分辨率范围 | 200x150 - 3072x4080 |
| 格式 | jpeg |

## 样例示例

**子集**: `Trajectory Reasoning`

```json
{
  "input": [
    {
      "id": "eec4c5cf",
      "content": [
        {
          "text": "If the yellow robot gripper follows the yellow trajectory, what will happen? Choices: A. Robot puts the soda on the wooden steps. B. Robot moves the soda in front of the wooden steps. C. Robot moves the soda to the very top of the wooden steps. D. Robot picks up the soda can and moves it up. Please answer directly with only the letter of the correct option and nothing else."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~16.0KB]"
        }
      ]
    }
  ],
  "choices": [
    "A",
    "B",
    "C",
    "D"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "Trajectory Reasoning",
  "metadata": {
    "question_id": "ERQA_1",
    "question_type": "Trajectory Reasoning"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets erqa \
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
    datasets=['erqa'],
    dataset_args={
        'erqa': {
            # subset_list: ['Trajectory Reasoning', 'Action Reasoning', 'Pointing']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```