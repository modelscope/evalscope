# General-VMCQ


## 概述

General-VMCQ 是一个可自定义的视觉多选题问答基准测试，用于评估多模态模型。它采用 MMMU 风格的格式，在文本中使用图像占位符，支持灵活的图像输入。

## 任务描述

- **任务类型**：视觉多选题问答（Visual Multiple-Choice Question Answering）
- **输入**：包含 `<image N>` 占位符的问题 + 选项 + 图像
- **输出**：所选答案选项
- **灵活性**：通过本地文件支持自定义数据集

## 主要特性

- 采用 MMMU 风格格式（非 OpenAI 消息格式）
- 每个样本最多支持 100 张图像
- 灵活的图像输入方式（路径、URL 或 base64 数据 URL）
- 支持思维链（Chain-of-thought）提示模板
- 通过本地文件加载支持自定义数据集

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率（Accuracy）**
- 训练集划分：**dev**，评估集划分：**val**
- 图像以纯字符串形式提供（不要包装成 `{{"url": ...}}` 格式）
- 数据集格式详情请参阅 [用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_vmcq` |
| **数据集ID** | `general_vmcq` |
| **论文** | N/A |
| **标签** | `Custom`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估集划分** | `val` |
| **训练集划分** | `dev` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets general_vmcq \
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
    datasets=['general_vmcq'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```