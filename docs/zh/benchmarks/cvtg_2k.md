# CVTG-2K


## 概述

CVTG-2K 评估文本到图像模型能否在单张图像中准确渲染多个文本区域。

## 任务描述

- **任务类型**：文本到图像生成
- **输入**：指定 2-5 个带引号文本区域的提示词
- **输出**：包含所请求文本的图像
- **领域**：多区域文本渲染

## 主要特点

- 来自 TextCrafter: CVTG (1,002) 和 CVTG-Style (998) 的 2,000 条提示词。
- CVTG-Style 额外指定了文本大小、颜色和字体。
- 默认加载 ModelScope 镜像；同时也支持官方 Hugging Face 源。

## 评估说明

- 安装 `evalscope[cvtg_2k]` 以使用 PaddleOCR 进行评分。请在支持 PaddlePaddle 2.6 的平台上的 Python 3.11 或 3.12 环境中运行。
- 单词准确率（Word Accuracy）和单词级归一化编辑相似度（NED，值越高越好）遵循官方评估器标准，对所有真实标签单词进行聚合计算，而非对图像得分取平均值。
- 这些指标仅评估文本内容，不涉及字体、颜色、布局或图像美学。
- 评分失败的样本将被排除；对于未识别出任何文本的有效图像，视为缺失文本。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cvtg_2k` |
| **数据集ID** | [evalscope/CVTG-2K](https://modelscope.cn/datasets/evalscope/CVTG-2K/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2503.23461) |
| **标签** | `TextToImage` |
| **指标** | `CVTGWordAccuracy`, `CVTGNED` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,000 |
| 提示词长度（平均） | 195.92 字符 |
| 提示词长度（最小/最大） | 67 / 462 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `CVTG` | 1,002 | 161.05 | 67 | 343 |
| `CVTG-Style` | 998 | 230.93 | 106 | 462 |

## 样例示例

**子集**: `CVTG`

```json
{
  "input": [
    {
      "id": "58a209ac",
      "content": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "prompt": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'.",
    "carrier_list": [
      "sign",
      "cup"
    ],
    "sentence_list": [
      "a sign that says 'READ'.",
      "a coffee cup on the table with the word 'MORNING'."
    ],
    "benchmark_type": "CVTG",
    "region_count": 2,
    "category": "CVTG",
    "tags": [],
    "id": "",
    "image_path": ""
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets cvtg_2k \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['cvtg_2k'],
    dataset_args={
        'cvtg_2k': {
            # subset_list: ['CVTG', 'CVTG-Style']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
