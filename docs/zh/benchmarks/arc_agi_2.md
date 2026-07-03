# ARC-AGI-2


## 概述

ARC-AGI-2（Abstraction and Reasoning Corpus for Artificial General Intelligence 2）是一个用于衡量 AI 系统能否仅凭少量示例就快速掌握新技能的基准测试。它通过网格变换任务来评估抽象推理和模式识别能力。

## 任务描述

- **任务类型**：抽象推理 / 模式识别  
- **输入**：一系列输入-输出网格对（示例），后跟一个测试输入网格  
- **输出**：根据推断出的变换规则生成的预测输出网格  
- **网格格式**：整数（0-9）组成的二维数组，尺寸可变（最大 30x30）

## 主要特点

- 包含 1,000 个公开训练任务和 120 个公开评估任务  
- 每个任务提供 2-10 个示例输入/输出对  
- 模型必须从示例中推断出变换规则  
- 测试的是抽象推理能力，而非依赖已学习的知识  
- 要求输出像素级精确（必须与目标网格完全一致）

## 评估说明

- 评分基于 **完全匹配**（形状和所有数值必须完全相同）  
- 模型必须以 JSON 二维数组格式输出网格  
- 零样本评估（每个任务内提供示例）  
- 设计为人类可解但对 AI 具有挑战性  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `arc_agi_2` |
| **数据集ID** | [evalscope/arc-agi-2](https://modelscope.cn/datasets/evalscope/arc-agi-2/summary) |
| **论文** | N/A |
| **标签** | `Reasoning` |
| **指标** | `acc` |
| **默认样本数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 120 |
| 提示词长度（平均） | 8026.79 字符 |
| 提示词长度（最小/最大） | 2437 / 25471 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "7b97143f",
      "content": "You are an expert at abstract reasoning and pattern recognition. Given input-output grid pairs as examples, you must figure out the transformation rule and apply it to a new test input to produce the correct output grid."
    },
    {
      "id": "bdaa2b22",
      "content": "You are given a series of input-output grid pairs as examples. Each grid is a 2D array of integers (0-9). Study the pattern in the examples, then predict the output for the test input.\n\nExamples:\nExample 1:\nInput: [[0, 0, 0, 0, 0, 0, 0, 0, 0, ... [TRUNCATED 7482 chars] ... 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]\n\nProvide the output grid as a JSON 2D array. Only output the JSON array, nothing else."
    }
  ],
  "target": "[[8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8], [8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [8, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 0], [ ... [TRUNCATED 1596 chars] ... ], [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 8, 0], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0]]",
  "id": 0,
  "group_id": 0
}
```

## 提示模板

**系统提示：**
```text
You are an expert at abstract reasoning and pattern recognition. Given input-output grid pairs as examples, you must figure out the transformation rule and apply it to a new test input to produce the correct output grid.
```

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
    --datasets arc_agi_2 \
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
    datasets=['arc_agi_2'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```