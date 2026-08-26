# VTCBench


## 概述

VTCBench (Vision-Text Compression Benchmark) 评估 VLM 压缩视觉文本的能力。

## 任务描述

- **任务类型**：具有双重评估模式的视觉问答
- **输入**：(VTC) 图像 + 问题文本，或 (Text) 文本上下文 + 问题文本
- **输出**：简短的自由格式答案
- **领域**：通用视觉理解、富含文本的图像理解

## 主要特性

- 双重评估模式：基于图像 (VTC) 和基于文本 (Text)
- VTC 模式通过直接输入图像来测试模型的视觉理解能力
- Text 模式利用图像的文本上下文来测试模型基于文本的推理能力
- 该差距突显了模型在回答问题时利用视觉信息与文本上下文的能力对比

## 评估说明

- 默认配置使用 **0-shot** 评估
- 长上下文基准测试需要更长的间隔，请将 `retry_interval` 设置得更高以避免超时
- 使用 `--dataset-args '{"vtcbench": {"extra_params":{"eval_mode":"text"}}}'` 切换模式，默认为 'vtc'
- 指标：
  - 检索和推理子集使用 **containsAll**/**ROUGE-1-R**
  - 记忆子集使用 **ROUGE-L-R**/**LLM-Judge**
- 如果遇到转换偏移溢出问题，请设置 `DATASET_TF_BATCH_SIZE=1`


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `vtcbench` |
| **数据集 ID** | [MLLM-CL/VTCBench](https://modelscope.cn/datasets/MLLM-CL/VTCBench/summary) |
| **论文** | N/A |
| **标签** | `LongContext`, `MultiModal`, `QA`, `Reasoning`, `Retrieval` |
| **指标** | `Rouge` |
| **默认 Shot 数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

*暂无统计数据。*

## 样例示例

*暂无样例示例。*

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `vtc` | 评估模式：vtc（图像 + 问题）或 text（文本 + 问题）。选项：['vtc', 'text'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vtcbench \
    --limit 10  # 正式评估时请移除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['vtcbench'],
    dataset_args={
        'vtcbench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```
