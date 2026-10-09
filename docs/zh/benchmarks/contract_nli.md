# ContractNLI (classification)


## Overview

对完整的保密协议（NDA）进行文档级自然语言推理。

## Task Description

- **任务类型**：单选分类
- **输入**：完整合同文本及假设语句
- **输出**：蕴含（Entailment）、矛盾（Contradiction）或未提及（Not mentioned）
- **领域**：法律文档理解

## Key Features

- 公开数据集：Hugging Face 上的 `tasksource/contract-nli`
- 保留原始标签和完整的任务上下文
- 支持聊天生成模型和文本 System One Choice 模型

## Evaluation Notes

- 使用完整文档的 contractnli_b 数据镜像（仅含数据）。每个合同-假设对构成一个样本。仅报告分类准确率；不评估证据抽取和官方 F1 指标。合同与假设的哈希值用于标识相关样本。
- 默认采用 0-shot 设置；若存在训练集，可配置训练示例
- 系统提示将转换为选择题任务指令，不使用原生的聊天角色层级结构
- 评估语义版本：v1.0


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `contract_nli` |
| **Dataset ID** | [tasksource/contract-nli](https://huggingface.co/datasets/tasksource/contract-nli) |
| **Paper** | N/A |
| **Tags** | `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,091 |
| Prompt Length (Mean) | 11719.39 chars |
| Prompt Length (Min/Max) | 1701 / 42345 chars |

## Sample Example

**Subset**: `contractnli_b`

```json
{
  "input": [
    {
      "id": "499ba6f1",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C.\n\nContract:\nNON-DISCLOSURE AGREEMENT\nRequired under JEA's ... [TRUNCATED 16664 chars] ... body Disclosing Party's Confidential Information.\n\nA) Contradiction: the contract contradicts the hypothesis.\nB) Entailment: the contract supports the hypothesis.\nC) Not mentioned: the contract neither supports nor contradicts the hypothesis."
    }
  ],
  "choices": [
    "Contradiction: the contract contradicts the hypothesis.",
    "Entailment: the contract supports the hypothesis.",
    "Not mentioned: the contract neither supports nor contradicts the hypothesis."
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "contract_id": "944f3cc63d215fdeb4182cbeb0d4b6ed3f3a5dcc51fe0680c879e709cabc0502",
    "hypothesis_id": "e1ce8a23b0a81e1223e43ac59ff5c4eae4389b816b2b08c8803b5c02c1f94eca"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets contract_nli \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['contract_nli'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
