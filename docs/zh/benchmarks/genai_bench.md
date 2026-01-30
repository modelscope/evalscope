# GenAI-Bench

## 概述

GenAI-Bench 是一个全面的文本到图像基准测试，包含 1600 个提示词，旨在评估图像生成模型在不同类别和复杂度下的表现。

## 任务描述

- **任务类型**：文本到图像生成评估  
- **输入**：复杂度各异的文本提示（基础与高级）  
- **输出**：使用 VQAScore 评估生成的图像  
- **规模**：1600 个提示词  

## 主要特性

- 大规模提示词集合，支持全面评估  
- 提示词按类别划分（基础 vs 高级）  
- 使用 VQAScore 衡量语义对齐程度  
- 包含丰富的元数据（如类别标签）  
- 支持对生成图像和已有图像进行评估  

## 评估说明

- 默认配置采用 **0-shot** 评估方式  
- 主要指标：**VQAScore**（用于衡量语义对齐）  
- 评估对象为 **test** 分割中的图像  
- 提示词根据复杂度分为“basic”或“advanced”两类  
- 属于 T2V-Eval-Prompts 数据集集合的一部分  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `genai_bench` |
| **数据集ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **论文** | N/A |
| **标签** | `TextToImage` |
| **指标** | `VQAScore` |
| **默认Shots数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,600 |
| 提示词长度（平均） | 67.42 字符 |
| 提示词长度（最小/最大） | 14 / 192 字符 |

## 样例示例

**子集**：`GenAI-Bench-1600`

```json
{
  "input": [
    {
      "id": "24dc1965",
      "content": "A baker pulling freshly baked bread out of an oven in a bakery."
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "GenAI-Bench1600_0",
    "prompt": "A baker pulling freshly baked bread out of an oven in a bakery.",
    "category": "basic",
    "tags": {
      "advanced": [],
      "basic": [
        "Attribute",
        "Scene",
        "Spatial Relation",
        "Action Relation"
      ]
    },
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
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets genai_bench \
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
    datasets=['genai_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```