# VQAv2


## 概览

VQAv2 是构建在 COCO 图像上的平衡版视觉问答基准。它评测多模态模型能否基于图像内容，
回答开放式自然语言问题。

## 任务描述

- **任务类型**：开放式视觉问答
- **输入**：图像 + 自然语言问题
- **输出**：简短答案短语
- **领域**：通用图像理解、物体识别、计数、属性、关系

## 评测说明

- 默认数据源：ModelScope 上的 `lmms-lab/VQAv2`，`validation` 划分
- 也可以通过设置 `extra_params.dataset_hub="huggingface"` 使用 Hugging Face
- 主要指标：基于人工标注答案的 **VQAv2 软准确率**
- 也会报告与可用答案集合之间的归一化精确匹配
- 适配器支持常见答案格式：字符串列表、答案字典列表，或 `multiple_choice_answer`


## 属性

| 属性 | 值 |
|----------|-------|
| **基准名称** | `vqav2` |
| **数据集 ID** | [lmms-lab/VQAv2](https://modelscope.cn/datasets/lmms-lab/VQAv2/summary) |
| **论文** | [论文](https://arxiv.org/abs/1612.00837) |
| **标签** | `MultiModal`, `QA` |
| **指标** | `vqa_score`, `exact_match` |
| **默认样本数** | 0-shot |
| **评测划分** | `validation` |


## 数据统计

*暂无统计信息。*

## 样例

*暂无样例。*

## Prompt 模板

**Prompt 模板：**
```text
Answer the question according to the image using a short phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes).
```

## 额外参数

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `dataset_hub` | `str` | `modelscope` | 用于加载 VQAv2 标注和图像的数据集平台。可选值：['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | 要加载的源数据划分；默认使用 validation。 |
| `dataset_revision` | `str` | `` | 可选的数据集版本；留空时使用平台默认版本。 |
| `image_dir` | `str` | `` | 可选的本地目录，包含本地 JSONL/CSV 数据使用的 VQAv2 图像。 |
| `image_extension` | `str` | `` | 可选的本地图像扩展名覆盖值，例如 "jpg"。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vqav2 \
    --limit 10  # 正式评测时移除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['vqav2'],
    dataset_args={
        'vqav2': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评测时移除此行
)

run_task(task_cfg=task_cfg)
```
