# VTCBench


## 概述

VTCBench（Vision-Text Compression Benchmark）评估将文本表示为渲染图像时的长上下文理解能力，并与纯文本基线进行比较。

## 任务描述

- **任务类型**：包含图像模式与文本模式的长上下文问答
- **输入**：渲染后的上下文图像及问题（VTC 模式），或原始文本及问题（Text 模式）
- **输出**：简短的自由格式答案
- **领域**：检索、关联推理和长期对话记忆

## 主要特性

- 提供配对的 VTC 和 Text 模式，用于衡量视觉文本压缩的影响
- 包含源自 RULER、NoLiMa 和 LoCoMo 的 Retrieval、Reasoning 和 Memory 子集
- 使用预渲染的多图文档，以保留基准测试的视觉布局
- 支持跨多张文档图像的上下文

## 评估说明

- 默认配置在 VTC 模式下使用 **0-shot** 评估
- 使用 `--dataset-args '{"vtcbench": {"extra_params":{"eval_mode":"text"}}}'` 启用 Text 基线
- Retrieval 和 Reasoning 使用官方的分数型 `contains_all` 指标
- Memory 使用各参考答案中最高的官方 ROUGE-L F1 分数
- 统一的 `score` 指标会根据子集选择相应的官方指标；报告中的 `macro_score` 是三个任务的无权平均值
- Text 模式按照官方静态评测器的方式移除 HTML 标签并规范化空白字符
- 评分前会排除 `<think>...</think>` 中的内容，与官方评测器保持一致
- 长上下文请求可能需要设置更大的模型超时时间
- 如果数据集转换出现 offset overflow，请设置 `DATASET_TF_BATCH_SIZE=1`
- [论文](https://arxiv.org/abs/2512.15649) | [代码](https://github.com/Moenupa/VTCBench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `vtcbench` |
| **数据集 ID** | [MLLM-CL/VTCBench](https://modelscope.cn/datasets/MLLM-CL/VTCBench/summary) |
| **论文** | [论文](https://arxiv.org/abs/2512.15649) |
| **标签** | `LongContext`, `MultiModal`, `QA`, `Reasoning`, `Retrieval` |
| **指标** | `score`, `contains_all`, `rouge_l` |
| **默认 Shot 数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 样本总数 | 2,200 |
| Prompt 平均长度 | 236.71 字符 |
| Prompt 最短/最长长度 | 89 / 384 字符 |

**各子集统计：**

| 子集 | 样本数 | Prompt 平均长度 | Prompt 最短长度 | Prompt 最长长度 |
|--------|---------|-------------|------------|------------|
| `Retrieval` | 800 | 110.38 | 89 | 141 |
| `Reasoning` | 800 | 368.69 | 363 | 384 |
| `Memory` | 600 | 229.15 | 186 | 283 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 26,554 |
| 每个样本的图像数 | 最少：1，最多：62，平均：12.07 |
| 分辨率范围 | 896x896 - 896x896 |
| 格式 | jpeg |


## 样例示例

**子集**：`Retrieval`

```json
{
  "input": [
    {
      "id": "c51f44e8",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~367.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~366.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~385.2KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~377.7KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~333.5KB]"
        },
        {
          "text": "\n\nQuestion:What are all the special magic numbers for foolish-rawhide mentioned in the provided text?"
        }
      ]
    }
  ],
  "target": "4075987, 5943250",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem": "What are all the special magic numbers for foolish-rawhide mentioned in the provided text?",
    "answers": [
      "4075987",
      "5943250"
    ],
    "subset": "Retrieval",
    "eval_mode": "vtc"
  }
}
```

## Prompt 模板

*未定义 Prompt 模板。*

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
            # subset_list: ['Retrieval', 'Reasoning', 'Memory']  # 可选：评估指定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```
