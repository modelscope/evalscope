# ContractNLI (classification)


## 概述

在完整的保密协议（NDA）上进行文档级自然语言推理。

## 任务描述

- **任务类型**：单选分类
- **输入**：完整合同文本和假设语句
- **输出**：蕴含（Entailment）、矛盾（Contradiction）或未提及（Not mentioned）
- **领域**：法律文档理解

## 主要特性

- 公开数据集：ModelScope 上的 `evalscope/contract-nli`
- 保留原始标签和完整的任务上下文
- 支持聊天生成模型和文本单选（System One Choice）模型

## 评估说明

- 使用完整文档的 contractnli_b 数据镜像（仅包含数据）。每个合同-假设对构成一个样本。仅报告分类准确率；不评估证据抽取和官方 F1 指标。合同和假设的哈希值用于标识相关样本。
- 默认采用 0-shot 设置；若存在训练集，可配置训练示例
- 系统提示将转换为单选任务指令，不使用原生的聊天角色层级结构
- 评估语义版本：v1.0

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `contract_nli` |
| **数据集ID** | [evalscope/contract-nli](https://modelscope.cn/datasets/evalscope/contract-nli/summary) |
| **论文** | N/A |
| **标签** | `MCQ` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估集** | `test` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,091 |
| 提示词长度（平均） | 11719.39 字符 |
| 提示词长度（最小/最大） | 1701 / 42345 字符 |

## 样例示例

**子集**: `contractnli_b`

```json
{
  "input": [
    {
      "id": "21deb5eb",
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

*注：部分内容为显示目的已截断。*

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
    --datasets contract_nli \
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
    datasets=['contract_nli'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
