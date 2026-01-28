# OCRBench-v2

## 概述

OCRBench v2 是一个大规模双语文本中心基准测试，包含最全面的 OCR 任务（比 OCRBench v1 多出 4 倍），涵盖 31 种多样化场景，包括街景、收据、公式、图表等。

## 任务描述

- **任务类型**：光学字符识别与文档理解
- **输入**：图像 + OCR/文档问题
- **输出**：文本识别、提取或分析结果
- **语言**：英语和中文（双语）

## 核心特性

- 10,000 个人工验证的问题-答案对
- 31 种多样化场景（街景、收据、公式、图表等）
- 高比例的困难样本
- 全面的 OCR 任务覆盖
- 支持双语（英语和中文）评估

## 评估说明

- 默认配置使用 **0-shot** 评估
- 在测试集上进行评估
- 依赖包：apted、distance、Levenshtein、lxml、Polygon3、zss
- 使用简单准确率（accuracy）作为评估指标

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ocr_bench_v2` |
| **数据集ID** | [evalscope/OCRBench_v2](https://modelscope.cn/datasets/evalscope/OCRBench_v2/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 10,000 |
| 提示词长度（平均） | 155.62 字符 |
| 提示词长度（最小/最大） | 6 / 1863 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `APP agent en` | 300 | 36.9 | 16 | 84 |
| `ASCII art classification en` | 200 | 174.62 | 162 | 193 |
| `key information extraction cn` | 400 | 32.45 | 9 | 162 |
| `key information extraction en` | 400 | 506.23 | 327 | 1261 |
| `key information mapping en` | 300 | 664.02 | 429 | 1647 |
| `VQA with position en` | 300 | 557.69 | 539 | 612 |
| `chart parsing en` | 400 | 67 | 67 | 67 |
| `cognition VQA cn` | 200 | 15.91 | 6 | 48 |
| `cognition VQA en` | 800 | 44.73 | 14 | 179 |
| `diagram QA en` | 300 | 118.23 | 54 | 356 |
| `document classification en` | 200 | 314 | 314 | 314 |
| `document parsing cn` | 300 | 43 | 43 | 43 |
| `document parsing en` | 400 | 51 | 51 | 51 |
| `formula recognition cn` | 200 | 19 | 19 | 19 |
| `formula recognition en` | 400 | 58.39 | 53 | 60 |
| `handwritten answer extraction cn` | 200 | 39.75 | 22 | 60 |
| `math QA en` | 300 | 119 | 119 | 119 |
| `full-page OCR cn` | 200 | 31 | 31 | 31 |
| `full-page OCR en` | 200 | 91 | 91 | 91 |
| `reasoning VQA en` | 600 | 80.45 | 26 | 256 |
| `reasoning VQA cn` | 400 | 163.96 | 62 | 633 |
| `fine-grained text recognition en` | 200 | 155.63 | 152 | 156 |
| `science QA en` | 300 | 387.54 | 159 | 1863 |
| `table parsing cn` | 300 | 139.4 | 79 | 211 |
| `table parsing en` | 400 | 65.39 | 57 | 134 |
| `text counting en` | 200 | 113.87 | 101 | 127 |
| `text grounding en` | 200 | 361.5 | 357 | 379 |
| `text recognition en` | 800 | 37.26 | 29 | 104 |
| `text spotting en` | 200 | 446 | 446 | 446 |
| `text translation cn` | 400 | 230.88 | 96 | 291 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 10,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 19x10 - 3912x21253 |
| 格式 | jpeg |

## 样例示例

**子集**: `APP agent en`

```json
{
  "input": [
    {
      "id": "51292e32",
      "content": [
        {
          "text": "What is the wrong answer 2?"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~121.7KB]"
        }
      ]
    }
  ],
  "target": "[\"enabled\", \"on\"]",
  "id": 0,
  "group_id": 0,
  "subset_key": "APP agent en",
  "metadata": {
    "question": "What is the wrong answer 2?",
    "answers": [
      "enabled",
      "on"
    ],
    "eval": "None",
    "dataset_name": "rico",
    "type": "APP agent en",
    "bbox": null,
    "bbox_list": null,
    "content": null
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ocr_bench_v2 \
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
    datasets=['ocr_bench_v2'],
    dataset_args={
        'ocr_bench_v2': {
            # subset_list: ['APP agent en', 'ASCII art classification en', 'key information extraction cn']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```