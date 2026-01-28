# General-QA


## 概述

General-QA 是一个可自定义的问题回答基准测试，用于评估语言模型在开放式文本生成任务上的表现。它支持灵活的数据格式和可配置的评估指标。

## 任务描述

- **任务类型**：开放式问答（Open-Ended Question Answering）
- **输入**：问题（可选系统提示和对话历史）
- **输出**：自由格式的文本答案
- **灵活性**：通过本地文件支持自定义数据集

## 主要特性

- 灵活的输入格式（query/answer 或 messages 格式）
- 可选的系统提示支持
- BLEU 和 Rouge 评估指标
- 通过本地文件加载支持自定义数据集
- 可扩展以适用于多种问答场景

## 评估说明

- 默认配置使用 **0-shot** 评估
- 默认指标：**BLEU**、**Rouge**（以 Rouge-L-R 为主评分）
- 在 **test** 数据划分上进行评估
- 数据集格式请参见 [用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_qa` |
| **数据集ID** | `general_qa` |
| **论文** | N/A |
| **标签** | `Custom`, `QA` |
| **指标** | `BLEU`, `Rouge` |
| **默认示例数（Shots）** | 0-shot |
| **评估数据划分** | `test` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
请回答问题
{question}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_qa \
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
    datasets=['general_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```