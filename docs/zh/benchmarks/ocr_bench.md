# OCRBench

## 概述

OCRBench 是一个综合评估基准，旨在评估大语言多模态模型的 OCR（光学字符识别）能力。它涵盖五个关键 OCR 相关任务，包含 1,000 个经过人工验证的问题-答案对。

## 任务描述

- **任务类型**：OCR 与文档理解  
- **输入**：包含 OCR 相关问题的图像  
- **输出**：文本识别或提取结果  
- **组成部分**：文本识别、视觉问答（VQA）、文档导向的 VQA、关键信息提取、数学表达式识别  

## 主要特点

- 覆盖 10 个类别的 1,000 个问答对  
- 答案经过人工验证和修正  
- 类别包括：常规/不规则/艺术字体/手写文本识别  
- 以场景文本为中心的 VQA 和面向文档的 VQA  
- 关键信息提取  
- 手写数学表达式识别（HME100k）

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 使用简单准确率指标（基于包含匹配）  
- 结果按问题类型/类别细分  
- HME100k 使用不同的匹配规则（忽略空格）  
- 全面测试多模态模型中的 OCR 能力  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ocr_bench` |
| **数据集 ID** | [evalscope/OCRBench](https://modelscope.cn/datasets/evalscope/OCRBench/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（平均） | 55.78 字符 |
| 提示词长度（最小/最大） | 14 / 149 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Regular Text Recognition` | 50 | 29 | 29 | 29 |
| `Irregular Text Recognition` | 50 | 29 | 29 | 29 |
| `Artistic Text Recognition` | 50 | 29 | 29 | 29 |
| `Handwriting Recognition` | 50 | 29 | 29 | 29 |
| `Digit String Recognition` | 50 | 32 | 32 | 32 |
| `Non-Semantic Text Recognition` | 50 | 29 | 29 | 29 |
| `Scene Text-centric VQA` | 200 | 34.6 | 14 | 101 |
| `Doc-oriented VQA` | 200 | 59 | 21 | 136 |
| `Key Information Extraction` | 200 | 101.58 | 86 | 149 |
| `Handwritten Mathematical Expression Recognition` | 100 | 79 | 79 | 79 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 1,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 25x16 - 4961x7016 |
| 格式 | jpeg |

## 样例示例

**子集**: `Regular Text Recognition`

```json
{
  "input": [
    {
      "id": "ff23a835",
      "content": [
        {
          "text": "what is written in the image?"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~1.2KB]"
        }
      ]
    }
  ],
  "target": "[\"CENTRE\"]",
  "id": 0,
  "group_id": 0,
  "subset_key": "Regular Text Recognition",
  "metadata": {
    "dataset": "IIIT5K",
    "question_type": "Regular Text Recognition"
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
    --datasets ocr_bench \
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
    datasets=['ocr_bench'],
    dataset_args={
        'ocr_bench': {
            # subset_list: ['Regular Text Recognition', 'Irregular Text Recognition', 'Artistic Text Recognition']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```