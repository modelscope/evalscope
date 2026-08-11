# MeasureBench


## 概述

MeasureBench 是一个全面的基准测试，用于评估视觉语言模型（VLMs）从测量仪器中读取数值的能力。它涵盖了 **真实世界照片** 和 **合成生成图像** 两大类，包含 4 种设计类别下的 26 种仪器类型。

## 任务描述

- **任务类型**：开放式视觉问答（仪器读数）
- **输入**：一张测量仪器的图像 + 一个读数问题
- **输出**：仪器当前的读数（数值或时间，带单位）
- **领域**：电流表、时钟、温度计、秤、速度表等共 26 种仪器类型

## 主要特性

- 总计 2,442 个样本，分为两个子集：real_world（1,272）和 synthetic_test（1,170）
- 26 种仪器类型，4 种设计类别（表盘式、数字式、模拟式、线性式）
- 接受围绕正确值的一个容差区间，而非要求完全精确匹配
- 对于时钟：通过多个有效区间处理 12 小时制与 24 小时制的歧义
- 单位识别与数值准确性分别进行评估

## 评估说明

- 默认子集：**real_world** 和 **synthetic_test**（视为独立子集）
- 主要指标：**Accuracy**（`accuracy`）—— ``all_correct``：数值 *和* 单位均正确
- 次要指标：**number_acc**（仅数值）、**unit_acc**（仅单位）
- 两种评估器：``interval_matching``（单一有效范围）和 ``multi_interval_matching``（例如时钟的上午/下午）
- 模型输出应在最后一行采用格式 ``Answer: <value> <unit>``
- 每个样本的元数据中记录了 ``image_type``；按类型划分的结果可在评审文件的 ``subset_key`` 列中查看，但无法通过 ``subset_list`` 单独选择
- [论文](https://arxiv.org/abs/2510.26865) | [GitHub](https://github.com/flageval-baai/MeasureBench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `measure_bench` |
| **数据集ID** | [evalscope/MeasureBench](https://modelscope.cn/datasets/evalscope/MeasureBench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2510.26865) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `accuracy`, `number_acc`, `unit_acc` |
| **默认示例数** | 0-shot |
| **评估子集** | `real_world` |


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
    --datasets measure_bench \
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
    datasets=['measure_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
