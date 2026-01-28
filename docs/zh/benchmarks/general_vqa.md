# General-VQA


## 概述

General-VQA 是一个可自定义的视觉问答（Visual Question Answering, VQA）基准测试，用于评估多模态模型。它支持 OpenAI 兼容的消息格式，并提供灵活的图像输入方式（本地路径、URL 或 base64 编码）。

## 任务描述

- **任务类型**：视觉问答（Visual Question Answering）
- **输入**：图像 + 以 OpenAI 聊天格式表示的问题
- **输出**：自由格式的文本答案
- **灵活性**：通过 TSV/JSONL 文件支持自定义数据集

## 主要特性

- 支持 OpenAI 兼容的消息格式
- 支持多种图像输入方式（本地路径、URL、base64）
- 使用 BLEU 和 Rouge 指标进行灵活评估
- 通过本地文件加载支持自定义数据集
- 可扩展以适用于各种 VQA 应用场景

## 评估说明

- 默认配置使用 **0-shot** 评估
- 默认指标：**BLEU**、**Rouge**（主评分指标为 Rouge-L-R）
- 在 **test** 数据划分上进行评估
- 数据集格式详见 [用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_vqa` |
| **数据集ID** | `general_vqa` |
| **论文** | N/A |
| **标签** | `Custom`, `MultiModal`, `QA` |
| **指标** | `BLEU`, `Rouge` |
| **默认示例数量** | 0-shot |
| **评估数据划分** | `test` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 通过命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_vqa \
    --limit 10  # 正式评估时请删除此行
```

### 通过 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['general_vqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```