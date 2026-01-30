# EvalMuse


## 概述

EvalMuse 是一个文本到图像的基准测试，使用 FGA-BLIP2Score 指标进行细粒度分析，以评估生成图像的质量和语义对齐程度。

## 任务描述

- **任务类型**：文本到图像生成评估
- **输入**：用于图像生成的文本提示
- **输出**：评估生成图像的质量和语义保真度
- **指标**：FGA-BLIP2Score（基于 BLIP-2 的细粒度分析）

## 主要特性

- 细粒度语义对齐评估
- 使用 BLIP-2 视觉-语言模型进行评分
- 同时评估图像质量和对提示的遵循程度
- 支持多样化的提示类别
- 提供客观、可复现的指标

## 评估说明

- 默认配置使用 **0-shot** 评估
- 仅支持 **FGA_BLIP2Score** 指标
- 评估 **test** 分割中的图像
- 可评估预生成的图像或生成新图像

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `evalmuse` |
| **数据集ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **论文** | N/A |
| **标签** | `TextToImage` |
| **指标** | `FGA_BLIP2Score` |
| **默认Shots** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 199 |
| 提示词长度（平均） | 61.05 字符 |
| 提示词长度（最小/最大） | 8 / 347 字符 |

## 样例示例

**子集**：`EvalMuse`

```json
{
  "input": [
    {
      "id": "f8b508f4",
      "content": "wide angle photograph at night, award winning interior design apartment, night outside, dark, moody dim faint lighting, wood panel walls, cozy and calm, fabrics, textiles, pillows, lamps, colorful copper brass accents, secluded, many light sources, lamps, hardwood floors, book shelf, couch, desk, plants"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "prompt": "wide angle photograph at night, award winning interior design apartment, night outside, dark, moody dim faint lighting, wood panel walls, cozy and calm, fabrics, textiles, pillows, lamps, colorful copper brass accents, secluded, many light sources, lamps, hardwood floors, book shelf, couch, desk, plants",
    "category": "",
    "tags": [
      "photograph (activity)",
      "night (attribute)",
      "interior design apartment (location)",
      "outside (location)",
      "dark (attribute)",
      "moody dim faint lighting (attribute)",
      "wood panel walls (object)",
      "cozy and calm (attribute)",
      "fabrics (material)",
      "textiles (material)",
      "pillows (object)",
      "lamps (object)",
      "colorful (attribute)",
      "copper brass accents (material)",
      "secluded (attribute)",
      "many light sources (attribute)",
      "hardwood floors (material)",
      "book shelf (object)",
      "couch (object)",
      "desk (object)",
      "plants (object)",
      "wide angle (attribute)"
    ],
    "id": "EvalMuse_0",
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
    --datasets evalmuse \
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
    datasets=['evalmuse'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```