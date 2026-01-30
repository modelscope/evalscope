# GEdit-Bench

## 概述

GEdit-Bench（Grounded Edit Benchmark）是一个基于真实世界使用场景的图像编辑基准测试。它通过基于大语言模型（LLM）的评判机制，对图像编辑模型在多种编辑任务上的表现进行全面评估。

## 任务描述

- **任务类型**：图像编辑评估
- **输入**：源图像 + 编辑指令
- **输出**：由 LLM 评判器评估后的编辑图像
- **语言**：英语（en）和中文（cn）

## 核心特性

- 真实世界的编辑场景（背景替换、颜色调整、风格迁移等）
- 11 类编辑任务
- 基于 LLM 的评估，衡量语义一致性和感知质量
- 支持英文和中文指令
- 综合评分：语义一致性（Semantic Consistency）、感知质量（Perceptual Quality）、总体得分（Overall）

## 评估说明

- 默认配置使用 **0-shot** 评估
- 在 **train** 划分上进行评估（包含测试样本）
- 评估指标：**语义一致性**（Semantic Consistency）、**感知相似性**（Perceptual Similarity，通过 LLM 评判）
- 总体得分：语义一致性与感知质量得分的几何平均值
- 通过 `extra_params['language']` 配置语言（en/cn）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gedit` |
| **数据集 ID** | [stepfun-ai/GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench/summary) |
| **论文** | N/A |
| **标签** | `ImageEditing` |
| **指标** | `Semantic Consistency`, `Perceptual Similarity` |
| **默认 Shots** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 606 |
| 提示词长度（平均） | 42.46 字符 |
| 提示词长度（最小/最大） | 11 / 158 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `background_change` | 40 | 50.2 | 29 | 158 |
| `color_alter` | 40 | 41.5 | 23 | 143 |
| `material_alter` | 40 | 40.8 | 18 | 60 |
| `motion_change` | 40 | 44.05 | 20 | 87 |
| `ps_human` | 70 | 34.17 | 16 | 89 |
| `style_change` | 60 | 46.27 | 20 | 116 |
| `subject-add` | 60 | 51.13 | 14 | 148 |
| `subject-remove` | 57 | 37.3 | 15 | 110 |
| `subject-replace` | 60 | 48.95 | 27 | 96 |
| `text_change` | 99 | 39.71 | 11 | 116 |
| `tone_transfer` | 40 | 36 | 21 | 63 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 606 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 384x640 - 416x672 |
| 格式 | png |

## 样例示例

**子集**: `background_change`

```json
{
  "input": [
    {
      "id": "4c309b59",
      "content": [
        {
          "text": "Change the background to a city street."
        },
        {
          "image": "[BASE64_IMAGE: png, ~495.7KB]"
        }
      ]
    }
  ],
  "id": 0,
  "group_id": 0,
  "subset_key": "background_change",
  "metadata": {
    "task_type": "background_change",
    "key": "4a7d36259ad94d238a6e7e7e0bd6b643",
    "instruction": "Change the background to a city street.",
    "instruction_language": "en",
    "input_image": "[BASE64_IMAGE: png, ~495.7KB]",
    "Intersection_exist": true,
    "id": "4a7d36259ad94d238a6e7e7e0bd6b643"
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `language` | `str` | `en` | 指令语言。选项：['en', 'cn']。选项：['en', 'cn'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gedit \
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
    datasets=['gedit'],
    dataset_args={
        'gedit': {
            # subset_list: ['background_change', 'color_alter', 'material_alter']  # 可选，评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```