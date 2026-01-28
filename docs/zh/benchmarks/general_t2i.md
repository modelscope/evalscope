# general_t2i

## 概述

General Text-to-Image 是一个可自定义的基准测试适配器，用于通过用户提供的提示词和图像来评估文本到图像生成模型。

## 任务描述

- **任务类型**：自定义文本到图像评估
- **输入**：用户提供的文本提示词
- **输出**：使用可配置指标评估生成的图像
- **灵活性**：支持本地提示词文件

## 主要特性

- 灵活的自定义评估框架
- 支持加载本地提示词文件
- 可配置的评估指标（默认：PickScore）
- 从文件路径自动推导子集名称
- 兼容预生成的图像

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**PickScore**（可配置）
- 支持自定义数据集路径和提示词文件
- 自动从文件路径提取子集名称
- 可通过 `image_path` 字段评估已有图像

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_t2i` |
| **数据集ID** | `general_t2i` |
| **论文** | N/A |
| **标签** | `Custom`, `TextToImage` |
| **指标** | `PickScore` |
| **默认Shots数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_t2i \
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
    datasets=['general_t2i'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```