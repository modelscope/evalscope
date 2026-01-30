# MuSR

## 概述

MuSR（Multistep Soft Reasoning，多步软推理）是一个通过基于叙事的问题来评估复杂推理能力的基准测试。它包含谋杀谜题、物品放置和团队分配等场景，要求模型进行多步推理。

## 任务描述

- **任务类型**：复杂推理（多项选择题）
- **输入**：包含问题和选项的叙事场景
- **输出**：正确答案的字母（A-F）
- **领域**：谋杀谜题、物品追踪、团队分配

## 主要特点

- 基于叙事的推理问题
- 需要多步逻辑推理
- 三个不同的推理领域
- 测试约束满足与演绎能力
- 上下文较长，需仔细推理

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用思维链（Chain-of-Thought, CoT）提示
- 包含三个子集：`murder_mysteries`、`object_placements`、`team_allocation`
- 使用简单准确率（accuracy）作为指标
- 是一个具有挑战性的基准，要求仔细阅读

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `musr` |
| **数据集ID** | [AI-ModelScope/MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 756 |
| 提示词长度（平均） | 4891.57 字符 |
| 提示词长度（最小/最大） | 2812 / 7537 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `murder_mysteries` | 250 | 5743.1 | 4056 | 7537 |
| `object_placements` | 256 | 5294.0 | 3735 | 7525 |
| `team_allocation` | 250 | 3627.93 | 2812 | 4351 |

## 样例示例

**子集**: `murder_mysteries`

```json
{
  "input": [
    {
      "id": "5ec1a7bd",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B. Think step by step before answering.\n\nIn an adrenaline inducing ... [TRUNCATED] ... and wronged, over and over, at the same sight. It was quite a sight. \n\nWinston, shuffling back to the station, was left with one thought - Looks like Mackenzie had quite an eventful week.\n\nWho is the most likely murderer?\n\nA) Mackenzie\nB) Ana"
    }
  ],
  "choices": [
    "Mackenzie",
    "Ana"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0
}
```

*注：部分内容为显示目的已截断。*

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
    --datasets musr \
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
    datasets=['musr'],
    dataset_args={
        'musr': {
            # subset_list: ['murder_mysteries', 'object_placements', 'team_allocation']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```