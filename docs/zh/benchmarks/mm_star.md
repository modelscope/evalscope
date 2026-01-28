# MMStar

## 概述

MMStar 是一个精英级的视觉不可或缺型多模态基准测试，旨在确保评估过程中真正依赖视觉信息。每个样本都经过精心筛选，必须依靠实际的视觉理解才能作答，从而最大限度地减少数据泄露，并测试高级多模态能力。

## 任务描述

- **任务类型**：视觉依赖型多项选择问答（Vision-Dependent Multiple-Choice QA）  
- **输入**：图像 + 需要视觉理解的多项选择题  
- **输出**：单个答案字母（A/B/C/D）  
- **领域**：感知、推理、数学、科学技术  

## 核心特性

- 确保视觉依赖性 —— 无图像则无法回答问题  
- 最小化训练语料中的数据泄露  
- 测试高级多模态推理能力  
- 六大类别：粗粒度感知、细粒度感知、实例推理、逻辑推理、数学、科学技术  
- 高质量人工筛选样本，经验证确需视觉信息  

## 评估说明

- 默认使用 **val** 划分进行评估  
- 主要指标：多项选择题的 **准确率（Accuracy）**  
- 使用思维链（Chain-of-Thought, CoT）提示，格式为 "ANSWER: [LETTER]"  
- 结果按类别和整体分别报告  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mm_star` |
| **数据集ID** | [evalscope/MMStar](https://modelscope.cn/datasets/evalscope/MMStar/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `val` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,500 |
| 提示词长度（平均） | 390.23 字符 |
| 提示词长度（最小/最大） | 272 / 2023 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `coarse perception` | 250 | 350.8 | 282 | 784 |
| `fine-grained perception` | 250 | 334.98 | 277 | 608 |
| `instance reasoning` | 250 | 379.02 | 273 | 684 |
| `logical reasoning` | 250 | 427.22 | 284 | 2023 |
| `math` | 250 | 467.36 | 292 | 891 |
| `science & technology` | 250 | 381.98 | 272 | 1173 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,500 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 114x66 - 3160x2136 |
| 格式 | jpeg |

## 样例示例

**子集**: `coarse perception`

```json
{
  "input": [
    {
      "id": "57e256b8",
      "content": [
        {
          "text": "Answer the following multiple choice question.\nThe last line of your response should be of the following format:\n'ANSWER: [LETTER]' (without quotes)\nwhere [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nWhich option describe the object relationship in the image correctly?\nOptions: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~37.2KB]"
        }
      ]
    }
  ],
  "choices": [
    "A",
    "B",
    "C",
    "D"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "coarse perception",
  "metadata": {
    "index": 0,
    "category": "coarse perception",
    "l2_category": "image scene and topic",
    "source": "MMBench",
    "split": "val",
    "image_path": "images/0.jpg"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question.
The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes)
where [LETTER] is one of A,B,C,D. Think step by step before answering.

{question}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mm_star \
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
    datasets=['mm_star'],
    dataset_args={
        'mm_star': {
            # subset_list: ['coarse perception', 'fine-grained perception', 'instance reasoning']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```