# ZeroBench

## 概述

ZeroBench 是一个面向大语言多模态模型（LMMs）的高难度视觉推理基准测试。它包含 100 个高质量、人工精心筛选的问题，涵盖多个领域、多种推理类型和图像类型，旨在超越当前模型的能力边界。

## 任务描述

- **任务类型**：高级视觉推理  
- **输入**：一张或多张图像 + 高难度视觉推理问题  
- **输出**：逐步推理过程，并在最后用花括号给出最终答案  
- **领域**：视觉推理、感知、多步推理  

## 核心特点

- 包含 100 个高质量、人工精心筛选的问题  
- 专为挑战前沿模型而设计（使用贪心解码时，零样本准确率为 0）  
- 覆盖多样化的领域、推理类型和图像类型  
- 目前尚无模型能达到 5/5 的可靠性评分  
- 用于测试当前视觉推理能力的极限  

## 评估说明

- 默认评估使用 **zerobench** 划分  
- 主要指标：由 LLM 评判的 **准确率（Accuracy）**  
- 答案格式必须为：`{final answer}`  
- 包含子问题划分，便于详细分析  
- 使用图像压缩以处理大尺寸图像  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `zerobench` |
| **数据集 ID** | [evalscope/zerobench](https://modelscope.cn/datasets/evalscope/zerobench/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `zerobench` |
| **训练划分** | `zerobench_subquestions` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 100 |
| 提示词长度（平均） | 645.72 字符 |
| 提示词长度（最小/最大） | 139 / 1998 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 108 |
| 每样本图像数 | 最小: 1, 最大: 3, 平均: 1.08 |
| 分辨率范围 | 512x297 - 5559x4070 |
| 图像格式 | jpeg, png |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "f3276b25",
      "content": [
        {
          "text": "I want to purchase all the Montellier bottles from the top three shelves. How much do I save by purchasing the bottles with a loyalty card? Give your final answer in dollars.\n\n\n\nLet's think step by step and give the final answer in curly braces,\nlike this: {final answer}\"\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~462.4KB]"
        }
      ]
    }
  ],
  "target": "11.90",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "1",
    "question_images": [
      "images/1_0.png"
    ],
    "image_attribution": "Own"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}



Let's think step by step and give the final answer in curly braces,
like this: {{final answer}}"

```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets zerobench \
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
    datasets=['zerobench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```