# Winogrande


## 概述

Winogrande 是一个大规模的常识推理基准测试，专门用于在 Winograd Schema Challenge 格式下测试代词消歧任务。它包含 4.4 万个问题，要求模型理解物理和社交方面的常识。

## 任务描述

- **任务类型**：代词消歧 / 常识推理
- **输入**：包含歧义代词的句子及两个选项
- **输出**：正确选项（A 或 B），用于消解代词指代
- **格式**：在两个名词短语之间进行二选一

## 主要特点

- 包含 4.4 万个 Winograd 风格的代词消歧问题
- 通过对抗性过滤减少数据集偏差
- 测试物理常识（物体属性、动作等）
- 测试社交常识（意图、情感等）
- 需要理解上下文以解决歧义

## 评估说明

- 默认配置使用 **0-shot** 评估
- 采用二选一格式（选项1 vs 选项2）
- 答案转换为 A/B 字母格式
- 使用简单准确率（accuracy）作为评估指标
- 常用于常识推理能力评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `winogrande` |
| **数据集ID** | [AI-ModelScope/winogrande_val](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,267 |
| 提示词长度（平均） | 306.58 字符 |
| 提示词长度（最小/最大） | 271 / 384 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "f64dd7e2",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nSarah was a much better surgeon than Maria so _ always got the easier cases.\n\nA) Sarah\nB) Maria"
    }
  ],
  "choices": [
    "Sarah",
    "Maria"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "winogrande_xs val 0"
  }
}
```

## 提示模板

**提示模板:**
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
    --datasets winogrande \
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
    datasets=['winogrande'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```