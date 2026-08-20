# BhashaBench-V1 (Krishi)


## 概述

BhashaBench-Krishi 是 BhashaBench-Multi 中 krishi（农业）领域的前身：这是一个领域特定的多项选择题基准测试，用于评估大语言模型在农业（Krishi）领域的知识，涵盖英语和印地语。

## 任务描述

- **任务类型**：领域特定的多项选择题问答
- **输入**：一道农业（Krishi）相关的问题，包含 4 个选项，语言为英语或印地语
- **输出**：正确答案对应的字母
- **语言**：英语、印地语

## 主要特点

- 每种语言包含 5,600 至 17,000 道题目，仅涵盖英语和印地语
- BhashaBench-Multi 的前身：领域相同，但语言覆盖范围更窄
- 每个领域对应一个独立的代码仓库，英语和印地语作为独立的配置项

## 评估说明

- 默认配置使用 **0-shot** 评估（仅提供 test 分割）
- 使用 `subset_list` 可评估单一语言（例如 `['Hindi']`）
- 需要访问此受限制的数据集：在 ModelScope（默认 Hub）上，请先接受条款并确保已登录；或者将 `dataset_hub` 设置为 `huggingface`，并在 huggingface.co 上接受条款后使用 `HF_TOKEN`
- 如需同一领域更广泛的语言覆盖，请参见 `bhasha_bench_multi_krishi`（涵盖 22 种印度语言，且无需授权）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bhashabenchv1_krishi` |
| **数据集ID** | [bharatgenai/BhashaBench-Krishi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Krishi/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 15,405 |
| 提示词长度（平均） | 409.45 字符 |
| 提示词长度（最小/最大） | 223 / 1841 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `English` | 12,648 | 429.18 | 223 | 1841 |
| `Hindi` | 2,757 | 318.93 | 233 | 678 |

## 样例示例

**子集**: `English`

```json
{
  "input": [
    {
      "id": "afa2a6e0",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nIt is state or condition of atmosphere at given place and given time.?\n\nA) Climate\nB) Weather\nC) Environment\nD) Atmosphere"
    }
  ],
  "choices": [
    "Climate",
    "Weather",
    "Environment",
    "Atmosphere"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English",
    "topic": ""
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

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bhashabenchv1_krishi \
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
    datasets=['bhashabenchv1_krishi'],
    dataset_args={
        'bhashabenchv1_krishi': {
            # subset_list: ['English', 'Hindi']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
