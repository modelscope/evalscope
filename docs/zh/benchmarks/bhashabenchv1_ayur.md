# BhashaBench-V1 (Ayurveda)


## 概述

BhashaBench-Ayur 是 BhashaBench-Multi 中 Ayurveda（阿育吠陀）领域的前身：这是一个领域特定的多项选择题基准测试，用于评估大语言模型对阿育吠陀医学知识的掌握程度，涵盖英语和印地语。

## 任务描述

- **任务类型**：领域特定的多项选择题问答
- **输入**：一道包含4个选项的阿育吠陀医学问题，语言为英语或印地语
- **输出**：正确答案对应的字母
- **语言**：英语、印地语

## 主要特点

- 每种语言包含5,600至17,000道题目，仅涵盖英语和印地语
- 作为 BhashaBench-Multi 的前身：领域相同，但语言覆盖范围更窄
- 每个领域对应一个独立的代码仓库，英语和印地语分别作为不同的配置

## 评估说明

- 默认配置使用 **0-shot** 评估（仅提供 test 分割）
- 使用 `subset_list` 可评估单一语言（例如 `['Hindi']`）
- 需要访问此受限制的数据集 —— 在 ModelScope（默认数据集中心）上，请先接受使用条款并确保已登录；或者将 `dataset_hub` 设置为 `huggingface`，并在 huggingface.co 上接受条款后使用 `HF_TOKEN`
- 如需同一领域但覆盖更多语言的版本，请参见 `bhasha_bench_multi_ayur`（涵盖22种印度语言，且无需授权）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bhashabenchv1_ayur` |
| **数据集ID** | [bharatgenai/BhashaBench-Ayur](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Ayur/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 14,963 |
| 提示词长度（平均） | 307.52 字符 |
| 提示词长度（最小/最大） | 222 / 1060 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `English` | 9,348 | 310.35 | 227 | 1060 |
| `Hindi` | 5,615 | 302.81 | 222 | 938 |

## 样例示例

**子集**: `English`

```json
{
  "input": [
    {
      "id": "6a6db8b7",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nImmune disorders include .....,\n\nA) Hypersensitivity\nB) auto-immune diseases\nC) immunodeficiency\nD) all of these"
    }
  ],
  "choices": [
    "Hypersensitivity",
    "auto-immune diseases",
    "immunodeficiency",
    "all of these"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English",
    "topic": "Kayachikitsa"
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
    --datasets bhashabenchv1_ayur \
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
    datasets=['bhashabenchv1_ayur'],
    dataset_args={
        'bhashabenchv1_ayur': {
            # subset_list: ['English', 'Hindi']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
