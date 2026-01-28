# VisuLogic


## 概述

VisuLogic 是一个用于评估多模态大语言模型（MLLMs）视觉推理能力的基准测试，其设计独立于文本推理。该基准包含精心构建的视觉推理任务，这些任务本质上难以仅通过语言进行描述。

## 任务描述

- **任务类型**：视觉推理（多项选择题）
- **输入**：图像 + 视觉推理问题（含4个选项）
- **输出**：答案字母（A/B/C/D）
- **领域**：纯视觉推理，不依赖基于文本的捷径

## 主要特点

- 六类推理技能：
  - **数量推理（Quantitative Reasoning）**：理解图像中数量的变化
  - **位置推理（Positional Reasoning）**：理解空间位置关系
  - **空间推理（Spatial Reasoning）**：理解三维空间关系
  - **属性推理（Attribute Reasoning）**：理解视觉属性
  - **风格推理（Stylistic Reasoning）**：理解视觉风格
  - **其他（Other）**：其他类型的视觉推理任务
- 测试模型真实的视觉理解能力，而非依赖语言捷径

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：多项选择题的 **准确率（Accuracy）**
- 使用思维链（Chain-of-Thought, CoT）提示，并要求以 "ANSWER: [LETTER]" 格式作答
- 结果按推理技能类别分组统计


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `visulogic` |
| **数据集ID** | [evalscope/VisuLogic](https://modelscope.cn/datasets/evalscope/VisuLogic/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（平均） | 394.16 字符 |
| 提示词长度（最小/最大） | 285 / 697 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Quantitative Reasoning` | 353 | 399.15 | 308 | 697 |
| `Other` | 108 | 399.08 | 285 | 560 |
| `Positional Reasoning` | 136 | 372.37 | 295 | 448 |
| `Stylistic Reasoning` | 90 | 375.32 | 303 | 483 |
| `Spatial Reasoning` | 231 | 401.91 | 314 | 537 |
| `Attribute Reasoning` | 82 | 401.15 | 295 | 458 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 288x125 - 700x825 |
| 格式 | jpeg, png |


## 样例示例

**子集**: `Quantitative Reasoning`

```json
{
  "input": [
    {
      "id": "1a6407d8",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.\n\nFrom the four given options, select the most suitable one to fill in the question mark, so that a certain regularity is presented:\n\n\n\nA: A  \nB: B  \nC: C  \nD: D"
        },
        {
          "image": "[BASE64_IMAGE: png, ~44.9KB]"
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
  "subset_key": "Quantitative Reasoning",
  "metadata": {
    "id": "00000"
  }
}
```

## 提示模板

**提示模板：**
```text

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}

```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets visulogic \
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
    datasets=['visulogic'],
    dataset_args={
        'visulogic': {
            # subset_list: ['Quantitative Reasoning', 'Other', 'Positional Reasoning']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```