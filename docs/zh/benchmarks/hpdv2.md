# HPD-v2

## 概述

HPD-v2（Human Preference Dataset v2）是一个文本到图像的基准测试，用于根据人类偏好评估生成的图像。它使用在大规模人类偏好数据上训练的 HPSv2.1 评分指标。

## 任务描述

- **任务类型**：文本到图像生成评估
- **输入**：用于图像生成的文本提示
- **输出**：根据人类偏好评估生成的图像
- **指标**：HPSv2.1 Score

## 主要特性

- 与人类偏好对齐的评估指标
- 在大规模人类偏好数据上训练
- 测试图像的美学质量和提示对齐程度
- 支持多样化的提示类别
- 客观、可复现的评分机制

## 评估说明

- 默认配置使用 **0-shot** 评估
- HPSv2.1 Score 指标衡量与人类偏好的对齐程度
- 支持本地提示文件
- 元数据中包含类别标签
- 可评估已有图像或生成新图像

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hpdv2` |
| **数据集ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **论文** | N/A |
| **标签** | `TextToImage` |
| **指标** | `HPSv2.1Score` |
| **默认 Shots** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,200 |
| 提示词长度（平均） | 81.71 字符 |
| 提示词长度（最小/最大） | 9 / 404 字符 |

## 样例示例

**子集**：`HPDv2`

```json
{
  "input": [
    {
      "id": "c971b3c7",
      "content": "Spongebob depicted in the style of Dragon Ball Z."
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "HPDv2_0",
    "prompt": "Spongebob depicted in the style of Dragon Ball Z.",
    "category": "Animation",
    "tags": {
      "category": "Animation"
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
    --datasets hpdv2 \
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
    datasets=['hpdv2'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```