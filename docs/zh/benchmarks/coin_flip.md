# CoinFlip

## 概述

CoinFlip 是一个符号推理基准测试，用于评估大语言模型（LLMs）在一系列操作中跟踪二元状态变化的能力。每个问题都要求判断一枚硬币在经历若干次翻转操作后的最终状态（正面/反面）。

## 任务描述

- **任务类型**：符号推理 / 状态追踪
- **输入**：不同人员执行的硬币翻转操作描述
- **输出**：硬币的最终状态（YES 表示正面朝上，NO 表示反面朝上）
- **重点**：二元状态追踪与逻辑推理

## 主要特点

- 测试模型在动作序列中追踪状态的能力
- 涉及二元推理（翻转/不翻转）决策
- 要求仔细关注操作者行为的影响
- 评估系统性的逻辑推理能力
- 答案清晰明确、无歧义

## 评估说明

- 默认配置使用 **0-shot** 评估
- 答案应遵循 "ANSWER: YES/NO" 格式
- 使用五个指标：准确率（accuracy）、精确率（precision）、召回率（recall）、F1 分数和 YES 比例（yes_ratio）
- 准确率是主要指标；其他指标用于辅助诊断
- 仅准确率以全部样本数为分母；若答案不是严格意义上的 YES/NO，则不会计入精确率、召回率和 F1 的计算，因此当答案格式错误较多时，这三个指标可能虚高
- 支持带推理示例的 few-shot 评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `coin_flip` |
| **数据集 ID** | [extraordinarylab/coin-flip](https://modelscope.cn/datasets/extraordinarylab/coin-flip/summary) |
| **论文** | N/A |
| **标签** | `Reasoning`, `Yes/No` |
| **指标** | `accuracy`, `precision`, `recall`, `f1`, `yes_ratio` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |
| **训练划分** | `validation` |


## 数据统计

*统计数据暂不可用。*

## 样例示例

*样例示例暂不可用。*

## 提示模板

**提示模板：**
```text

请逐步解决以下硬币翻转问题。你的回答最后一行必须是 "ANSWER: [ANSWER]"（不含引号）的形式，其中 [ANSWER] 是问题的答案。

{question}

请记住，在回答末尾单独一行写出答案，格式为 "ANSWER: [ANSWER]"（不含引号），其中 [ANSWER] 是 YES 或 NO。

推理过程：

```

<details>
<summary>Few-shot 模板</summary>

```text
以下是解决类似问题的一些示例：

{fewshot}


请逐步解决以下硬币翻转问题。你的回答最后一行必须是 "ANSWER: [ANSWER]"（不含引号）的形式，其中 [ANSWER] 是问题的答案。

{question}

请记住，在回答末尾单独一行写出答案，格式为 "ANSWER: [ANSWER]"（不含引号），其中 [ANSWER] 是 YES 或 NO。

推理过程：

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
