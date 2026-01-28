# GSM8K-V

## 概述

GSM8K-V 是一个纯视觉的多图像数学推理基准测试，它将每个 GSM8K 数学文字题系统性地转化为其视觉对应形式。该基准支持在多模态数学评估中对同一问题的不同模态进行清晰的对比。

## 任务描述

- **任务类型**：视觉数学文字题求解
- **输入**：表示数学问题的多张图像 + 问题
- **输出**：以 \boxed{} 格式给出的数值答案
- **领域**：视觉数学推理、场景理解、算术

## 主要特点

- 将 GSM8K 文本问题转化为视觉形式
- 采用多图像输入格式完整呈现问题
- 支持文本与视觉模态之间的对比
- 按问题类型和子类别进行分类
- 保留原始问题的难度等级

## 评估说明

- 默认评估使用 **train** 划分
- 主要指标：**准确率（Accuracy）**，通过数值比较计算
- 答案应仅包含数字，并置于 \boxed{} 中
- 鼓励在最终答案前提供逐步推理过程
- 元数据包含原始文本问题以供对比

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gsm8k_v` |
| **数据集ID** | [evalscope/GSM8K-V](https://modelscope.cn/datasets/evalscope/GSM8K-V/summary) |
| **论文** | N/A |
| **标签** | `Math`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,319 |
| 提示词长度（平均） | 492.4 字符 |
| 提示词长度（最小/最大） | 410 / 789 字符 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 5,344 |
| 每样本图像数 | 最小: 2, 最大: 11, 平均: 4.05 |
| 分辨率范围 | 1024x1024 - 1024x1024 |
| 格式 | jpeg |

## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "9e0072fd",
      "content": [
        {
          "text": "You are an expert at solving mathematical word problems. Please solve the following problem step by step, showing your reasoning.\n\nWhen providing your final answer:\n- If the answer can be expressed as a whole number (integer), provide it as a ... [TRUNCATED] ... s Janet (white woman, shoulder-length brown hair, wearing a light blue blouse and khaki pants) make each day at the farmers' market?\n\nPlease think step by step. After your reasoning, put your final answer within \\boxed{} with the number only."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~153.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~175.1KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~124.7KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~172.8KB]"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~117.4KB]"
        }
      ]
    }
  ],
  "target": "18.0",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 1,
    "category": "other",
    "subcategory": "count",
    "original_question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
You are an expert at solving mathematical word problems. Please solve the following problem step by step, showing your reasoning.

When providing your final answer:
- If the answer can be expressed as a whole number (integer), provide it as an integer
Problem: {question}

Please think step by step. After your reasoning, put your final answer within \boxed{} with the number only.

```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gsm8k_v \
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
    datasets=['gsm8k_v'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```