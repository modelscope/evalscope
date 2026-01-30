# HallusionBench

## 概述

HallusionBench 是一个高级诊断基准测试，旨在评估大视觉语言模型（LVLMs）的图像上下文推理能力，并检测其产生幻觉的倾向。该基准特别测试模型在语言幻觉和视觉错觉方面的敏感性。

## 任务描述

- **任务类型**：幻觉检测与视觉推理  
- **输入**：图像 + 关于图像内容的是/否问题  
- **输出**：YES 或 NO 答案  
- **领域**：幻觉检测、视觉推理、事实准确性  

## 主要特点

- 专门设计用于探测幻觉行为  
- 同时测试语言幻觉和视觉错觉  
- 按类别和子类别组织，便于详细分析  
- 使用分组准确率指标进行稳健评估  
- 问题要求精确的图像上下文推理  

## 评估说明

- 默认评估使用 **image** 切分  
- 多种准确率指标：
  - **aAcc**：答案级准确率（每题）
  - **fAcc**：图像级准确率（每张图像的所有问题均正确）
  - **qAcc**：问题级准确率（按问题类型分组）
- 要求仅输出 YES/NO，无需解释  
- 在子类别、类别和整体层面进行聚合  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hallusion_bench` |
| **数据集ID** | [lmms-lab/HallusionBench](https://modelscope.cn/datasets/lmms-lab/HallusionBench/summary) |
| **论文** | N/A |
| **标签** | `Hallucination`, `MultiModal`, `Yes/No` |
| **指标** | `aAcc`, `qAcc`, `fAcc` |
| **默认示例数** | 0-shot |
| **评估切分** | `image` |
| **聚合方式** | `f1` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 951 |
| 提示词长度（平均） | 136.78 字符 |
| 提示词长度（最小/最大） | 76 / 292 字符 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 951 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 388x56 - 5291x4536 |
| 格式 | png |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "ba75d669",
      "content": [
        {
          "text": "Is China, Hongkong SAR, the leading importing country of gold, silverware, and jewelry with the highest import value in 2018?\nPlease answer YES or NO without an explanation."
        },
        {
          "image": "[BASE64_IMAGE: png, ~143.0KB]"
        }
      ]
    }
  ],
  "target": "NO",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "category": "VS",
    "subcategory": "chart",
    "visual_input": "1",
    "set_id": "0",
    "figure_id": "1",
    "question_id": "0",
    "gt_answer": "0",
    "gt_answer_details": "Switzerland is the leading importing country of gold, silverware, and jewelry with the highest import value in 2018?"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
Please answer YES or NO without an explanation.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hallusion_bench \
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
    datasets=['hallusion_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```