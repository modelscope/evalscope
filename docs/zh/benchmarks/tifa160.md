# TIFA-160

## 概述

TIFA-160 是一个文本到图像的基准测试，包含 160 个精心策划的提示词，旨在通过基于视觉问答（VQA）的自动化评估方法，衡量生成图像的忠实度与质量。

## 任务描述

- **任务类型**：文本到图像生成评估  
- **输入**：用于图像生成的文本提示  
- **输出**：使用 PickScore 指标评估生成的图像  
- **规模**：160 个提示词  

## 主要特性

- 精简且高质量的提示词集合，支持高效评估  
- 使用 PickScore 指标对齐人类偏好  
- 覆盖多样化的图像生成能力  
- 支持评估新生成图像和已有图像  
- 提供可复现的评估流程  

## 评估说明

- 默认配置采用 **0-shot** 评估方式  
- 主要指标：**PickScore**（用于对齐人类偏好）  
- 评估对象来自 **test** 数据划分  
- 属于 T2V-Eval-Prompts 数据集集合的一部分  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tifa160` |
| **数据集 ID** | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) |
| **论文** | N/A |
| **标签** | `TextToImage` |
| **指标** | `PickScore` |
| **默认样本数（Shots）** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 160 |
| 提示词长度（平均） | 56.13 字符 |
| 提示词长度（最小/最大） | 13 / 182 字符 |

## 样例示例

**子集**：`TIFA-160`

```json
{
  "input": [
    {
      "id": "9de3e3b1",
      "content": "A Christmas tree with lights and teddy bear"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "prompt": "A Christmas tree with lights and teddy bear",
    "category": "",
    "tags": {},
    "id": "TIFA160_0",
    "image_path": ""
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tifa160 \
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
    datasets=['tifa160'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```