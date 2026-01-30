# General-MCQ


## 概述

General-MCQ 是一个可自定义的多项选择题问答基准测试，用于评估语言模型。它支持灵活的数据格式和可变数量的答案选项。

## 任务描述

- **任务类型**：多项选择题问答
- **输入**：包含 2 到 10 个答案选项（A 到 J）的问题
- **输出**：所选的答案选项
- **灵活性**：通过本地文件支持自定义数据集

## 主要特性

- 灵活的选项数量（A 到 J）
- 通过本地文件加载支持自定义数据集
- 中文单选题提示模板
- 可配置的少样本（few-shot）示例
- 基于准确率的评估

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率（Accuracy）**
- 训练集划分：**dev**，评估集划分：**val**
- 数据集格式详见 [用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#mcq)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_mcq` |
| **数据集ID** | `general_mcq` |
| **论文** | 无 |
| **标签** | `Custom`, `MCQ` |
| **指标** | `acc` |
| **默认样本数（Shots）** | 0-shot |
| **评估集划分** | `val` |
| **训练集划分** | `dev` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_mcq \
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
    datasets=['general_mcq'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```