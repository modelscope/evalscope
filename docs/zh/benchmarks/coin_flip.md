# CoinFlip

## 概述

CoinFlip 是一个符号推理基准测试，用于评估大语言模型（LLMs）在一系列操作中跟踪二元状态变化的能力。每个问题都涉及确定在多次翻转操作后硬币的最终状态（正面/反面）。

## 任务描述

- **任务类型**：符号推理 / 状态追踪
- **输入**：不同人员执行的硬币翻转操作描述
- **输出**：硬币的最终状态（YES 表示正面朝上，NO 表示反面朝上）
- **重点**：二元状态追踪与逻辑推理

## 主要特点

- 测试通过操作序列进行状态追踪的能力
- 二元推理（翻转/不翻转）决策
- 要求仔细关注操作者行为的影响
- 评估系统性逻辑推理能力
- 答案清晰明确、无歧义

## 评估说明

- 默认配置使用 **0-shot** 评估
- 答案应遵循 "ANSWER: YES/NO" 格式
- 采用五项指标：准确率（accuracy）、精确率（precision）、召回率（recall）、F1 分数（F1 score）和 YES 比例（yes_ratio）
- F1 分数为主要聚合指标
- 支持带推理示例的 few-shot 评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `coin_flip` |
| **数据集 ID** | [extraordinarylab/coin-flip](https://modelscope.cn/datasets/extraordinarylab/coin-flip/summary) |
| **论文** | N/A |
| **标签** | `Reasoning`, `Yes/No` |
| **指标** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **默认示例数量** | 0-shot |
| **评估分割** | `test` |
| **训练分割** | `validation` |
| **聚合方式** | `f1` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,333 |
| 提示词长度（平均） | 500.15 字符 |
| 提示词长度（最小/最大） | 453 / 551 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "05503706",
      "content": [
        {
          "text": "\nSolve the following coin flip problem step by step. The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem.\n\nQ: A coin is heads up. rushawn flips the coin. yerania ... [TRUNCATED] ...  the coin. jostin does not flip the coin.  Is the coin still heads up?\n\nRemember to put your answer on its own line at the end in the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer YES or NO to the problem.\n\nReasoning:\n"
        }
      ]
    }
  ],
  "target": "NO",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "NO"
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text

Solve the following coin flip problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer YES or NO to the problem.

Reasoning:

```

<details>
<summary>Few-shot 模板</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}


Solve the following coin flip problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer YES or NO to the problem.

Reasoning:

```

</details>

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets coin_flip \
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
    datasets=['coin_flip'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```