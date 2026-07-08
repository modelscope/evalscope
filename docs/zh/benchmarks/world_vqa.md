# WorldVQA


## 概述

WorldVQA 是一个用于评估多模态大语言模型（MLLMs）原子级视觉世界知识的基准测试。它衡量模型在分层分类体系中对视觉实体进行定位和命名的能力，涵盖从常见的头部类别物体到长尾稀有类别的广泛范围。

## 任务描述

- **任务类型**：视觉实体识别 / 知识问答（QA）
- **输入**：图像 + 要求识别视觉实体的问题
- **输出**：自由文本答案（具体实体名称）
- **领域**：自然、建筑、文化、产品、交通、娱乐、品牌、体育

## 主要特点

- 包含 3000 个 VQA 对，覆盖 8 个语义类别
- 双语支持：英语（non-zh）和中文（zh）
- 三个难度等级：简单、中等、困难
- 测试与推理解耦的原子级视觉知识
- 要求精确识别实体（例如，具体品种，而非泛称“狗”）

## 评估说明

- 默认配置采用 **0-shot** 评估
- 在 **train** 划分上进行评估（即基准测试数据划分）
- 主要指标：通过 LLM-as-judge 计算的 **准确率（Accuracy）**
- 支持使用 LLM 判定语义等价性
- 结果按类别及整体分别报告

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `world_vqa` |
| **数据集ID** | [evalscope/WorldVQA](https://modelscope.cn/datasets/evalscope/WorldVQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,000 |
| 提示词长度（平均） | 38.4 字符 |
| 提示词长度（最小/最大） | 5 / 182 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Nature & Environment` | 326 | 33.9 | 7 | 83 |
| `Locations & Architecture` | 512 | 42.03 | 9 | 122 |
| `Culture, Arts & Crafts` | 506 | 33.71 | 7 | 112 |
| `Objects & Products` | 437 | 55.97 | 9 | 182 |
| `Vehicles, Craft & Transportation` | 306 | 30.67 | 8 | 114 |
| `Entertainment, Media & Gaming` | 511 | 30.5 | 5 | 83 |
| `Brands, Logos & Graphic Design` | 260 | 39.1 | 6 | 83 |
| `Sports, Gear & Venues` | 142 | 41.99 | 9 | 100 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 3,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 33x65 - 3840x3840 |
| 格式 | gif, jpeg, png, webp |


## 样例示例

**子集**: `Nature & Environment`

```json
{
  "input": [
    {
      "id": "0c08c4e2",
      "content": [
        {
          "text": "What breed of dog is in the picture?"
        },
        {
          "image": "[BASE64_IMAGE: png, ~392.9KB]"
        }
      ]
    }
  ],
  "target": "Greek Hound",
  "id": 0,
  "group_id": 0,
  "subset_key": "Nature & Environment",
  "metadata": {
    "index": 0,
    "category": "Nature & Environment",
    "language": "non-zh",
    "difficulty": "medium"
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
    --datasets world_vqa \
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
    datasets=['world_vqa'],
    dataset_args={
        'world_vqa': {
            # subset_list: ['Nature & Environment', 'Locations & Architecture', 'Culture, Arts & Crafts']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
