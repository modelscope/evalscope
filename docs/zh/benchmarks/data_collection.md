# 数据收集（Data-Collection）

## 概述

Data-Collection 是一个灵活的框架，用于将多个评估数据集混合成统一的评估套件。它能够通过从各类基准测试中精心挑选的样本，对模型进行全面评估。

## 任务描述

- **任务类型**：多数据集统一评估  
- **输入**：来自多个基准数据集的混合样本  
- **输出**：跨任务、数据集和类别的聚合得分  
- **灵活性**：支持自定义数据集组合  

## 核心特性

- 将多个基准测试混合为单一评估  
- 分层报告（子集、数据集、任务、标签、类别层级）  
- 支持样本级权重  
- 为每个数据集自动初始化适配器  
- 全面的聚合方式（微观平均、宏观平均、加权平均）  

## 评估说明

- 数据集必须预先编译为集合形式  
- 支持多种任务类型（选择题 MCQ、问答 QA、代码生成等）  
- 生成多层级报告以支持详细分析  
- 使用方法请参阅 [Collection Guide](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `data_collection` |
| **数据集ID** | N/A |
| **论文** | N/A |
| **标签** | `Custom` |
| **指标** | `acc` |
| **默认示例数量（Shots）** | 0-shot |
| **评估划分** | `test` |

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
    --datasets data_collection \
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
    datasets=['data_collection'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```