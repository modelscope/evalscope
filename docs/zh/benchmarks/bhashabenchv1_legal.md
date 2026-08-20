# BhashaBench-V1 (Legal)


## 概述

BhashaBench-Legal 是 BhashaBench-Multi 法律领域的前身：这是一个领域特定的多项选择题基准测试，用于评估大语言模型对印度法律的知识，涵盖英语和印地语。

## 任务描述

- **任务类型**：领域特定的多项选择题问答
- **输入**：一道包含4个选项的印度法律问题，语言为英语或印地语
- **输出**：正确答案的字母
- **语言**：英语、印地语

## 主要特点

- 每种语言包含5,600至17,000道题目，仅涵盖英语和印地语
- 作为 BhashaBench-Multi 的前身：领域相同，但语言覆盖范围更窄
- 每个领域对应一个独立的代码仓库，英语和印地语分别作为独立的配置项

## 评估说明

- 默认配置使用 **0-shot** 评估（仅提供 test 分割）
- 使用 `subset_list` 可评估单一语言（例如 `['Hindi']`）
- 需要访问此受限制的数据集——在 ModelScope（默认 Hub）上，请先接受条款并确保已登录；或者将 `dataset_hub` 设置为 `huggingface`，并在 huggingface.co 上接受条款后使用 `HF_TOKEN`
- 如需同一领域但更广泛的语言覆盖，请参见 `bhasha_bench_multi_legal`（涵盖22种印度语言，无需授权）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bhashabenchv1_legal` |
| **数据集ID** | [bharatgenai/BhashaBench-Legal](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Legal/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 24,365 |
| 提示词长度（平均） | 513.88 字符 |
| 提示词长度（最小/最大） | 229 / 4628 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `English` | 17,047 | 539.36 | 233 | 4628 |
| `Hindi` | 7,318 | 454.52 | 229 | 1748 |

## 样例示例

**子集**: `English`

```json
{
  "input": [
    {
      "id": "6e1ae42b",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nPower to amend the issue or frame additional issues prior to passing of a decree vests in a Court by virtue of which provision of the Code of Civil Procedure, 1908?\n\nA) Order XIV Rule 1\nB) Order XIV Rule 5\nC) Order XIV Rule 6\nD) Section 151"
    }
  ],
  "choices": [
    "Order XIV Rule 1",
    "Order XIV Rule 5",
    "Order XIV Rule 6",
    "Section 151"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English",
    "topic": "Procedural Law"
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
    --datasets bhashabenchv1_legal \
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
    datasets=['bhashabenchv1_legal'],
    dataset_args={
        'bhashabenchv1_legal': {
            # subset_list: ['English', 'Hindi']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
