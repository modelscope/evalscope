# HellaSwag


## 概述

HellaSwag 是一个用于评估常识性自然语言推理的基准测试，专门测试模型完成描述日常情境句子的能力。该数据集采用对抗性过滤（adversarial filtering）方法生成具有挑战性的干扰项，这些干扰项在语法上正确但在语义上不合理。

## 任务描述

- **任务类型**：多项选择题句子补全
- **输入**：描述某项活动或情境的上下文
- **输出**：从 4 个选项（A、B、C、D）中选择最合理的后续内容
- **领域**：日常活动与常识场景

## 主要特点

- 包含 70,000 多个问题，用于测试基于现实情境的常识推理能力
- 上下文来源于 ActivityNet 和 WikiHow
- 错误选项经过对抗性过滤处理
- 要求理解典型事件序列
- 测试物理常识与社会常识推理能力

## 评估说明

- 默认配置使用 **0-shot** 评估方式
- 在验证集（validation split）上进行评估
- 已对选项结尾进行预处理，清除格式化伪影
- 上下文由 `ctx_a` 和 `ctx_b` 字段拼接而成
- 元数据中包含活动标签，可用于分析

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hellaswag` |
| **数据集ID** | [evalscope/hellaswag](https://modelscope.cn/datasets/evalscope/hellaswag/summary) |
| **论文** | N/A |
| **标签** | `Commonsense`, `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 10,042 |
| 提示词长度（平均） | 767.6 字符 |
| 提示词长度（最小/最大） | 329 / 1655 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "18df47e2",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nA man is sitting on a roof. He\n\nA) is using wrap to wrap a pair of skis.\nB) is ripping level tiles off.\nC) is holding a rubik's cube.\nD) starts pulling up roofing on a roof."
    }
  ],
  "choices": [
    "is using wrap to wrap a pair of skis.",
    "is ripping level tiles off.",
    "is holding a rubik's cube.",
    "starts pulling up roofing on a roof."
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "activity_label": "Roof shingle removal"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets hellaswag \
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
    datasets=['hellaswag'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```