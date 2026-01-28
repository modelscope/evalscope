# V*Bench

## 概述

V*Bench 是一个用于评估多模态推理系统中视觉搜索能力的基准测试。它专注于在高分辨率图像中主动定位和识别特定的视觉信息，这对于细粒度的视觉理解至关重要。

## 任务描述

- **任务类型**：视觉搜索与推理（多项选择题）
- **输入**：高分辨率图像 + 针对性的视觉查询
- **输出**：答案选项字母（A/B/C/D）
- **领域**：视觉搜索、细粒度识别、视觉定位（Visual Grounding）

## 主要特点

- 测试针对性视觉查询能力
- 聚焦高分辨率图像理解
- 要求查找并推理特定视觉元素
- 问题由自然语言指令引导
- 在复杂场景中评估细粒度视觉理解能力

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：多项选择题的 **准确率（Accuracy）**
- 使用思维链（Chain-of-Thought, CoT）提示，并以 "ANSWER: [LETTER]" 格式输出答案
- 元数据包含类别和问题 ID，便于分析

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `vstar_bench` |
| **数据集ID** | [lmms-lab/vstar-bench](https://modelscope.cn/datasets/lmms-lab/vstar-bench/summary) |
| **论文** | N/A |
| **标签** | `Grounding`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 191 |
| 提示词长度（平均） | 366.07 字符 |
| 提示词长度（最小/最大） | 332 / 402 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 191 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 1690x1500 - 5759x1440 |
| 格式 | jpeg |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "d0b9c304",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.\n\nWhat is the material of the glove?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather\nAnswer with the option's letter from the given choices directly."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~1.2MB]"
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
  "metadata": {
    "category": "direct_attributes",
    "question_id": "0"
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

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vstar_bench \
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
    datasets=['vstar_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```