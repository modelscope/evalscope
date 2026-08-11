# KINA


## 概述

KINA（Knowledge Index of Noah's Ark，诺亚方舟知识指数）是一个高密度的多学科知识基准测试，用于评估大语言模型能否解答涵盖261个细粒度学科的专家级问题。它是首个将“学科代表性”作为核心设计原则的基准测试。

## 任务描述

- **任务类型**：多项选择题问答（MCQ）
- **输入**：一个特定学科的问题，附带最多10个带字母编号的选项（A–J）
- **输出**：一个正确答案字母（A–J）
- **领域范围**：涵盖农学、医学、工程学、人文学科、自然科学等共计261个学科

## 主要特点

- 包含899道测试题，覆盖261个细粒度学科
- 每道题在最多10个选项（A–J）中仅有一个正确答案
- 提供每个选项的解释，用于训练或分析（不对模型展示）
- 旨在测试深层领域知识，而非检索能力或常识推理
- 在2077AI首次发布，强调学科代表性

## 评估说明

- 默认评估使用 **test** 划分（899个样本）
- 主要指标：**准确率**（`accuracy`）——单次推理模式下的 Pass@1
- 采用零样本思维链（0-shot Chain-of-Thought, CoT）评估方式，从 ``ANSWER: [LETTER]`` 标记中提取答案
- 每个样本均包含学科元数据，并可在评估结果中获取；但未按学科划分子集
- [GitHub](https://github.com/weihao1115/KINA-Benchmark)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `kina` |
| **数据集ID** | [evalscope/KINA](https://modelscope.cn/datasets/evalscope/KINA/summary) |
| **论文** | [Paper](https://www.2077ai.com/kina) |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `accuracy` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `test` |


## 数据统计

*统计数据暂不可用。*

## 样例示例

*样例示例暂不可用。*

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
    --datasets kina \
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
    datasets=['kina'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
