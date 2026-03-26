# MIA-Bench


## 概述

MIA-Bench 是一个多模态指令遵循基准测试，旨在评估视觉-语言模型在图像基础上遵循复杂、组合式指令的能力。每个样本包含一张图像和一条多组件指令，模型的响应由大语言模型（LLM）评委按组件分别打分。

## 任务描述

- **任务类型**：多模态指令遵循
- **输入**：图像 + 多组件指令
- **输出**：自由格式的回答，需满足所有指令组件的要求
- **领域**：视觉理解、指令遵循、语言生成

## 主要特点

- 包含 400 个测试样本，涵盖从基础到高级的多样化指令类型
- 每条指令被分解为 1–5 个评分组件，并配有加权分数
- 组件类型包括：描述（describe）、长度限制（length_limit）、语言学要求（linguistics）、格式（format）等
- 采用 LLM 作为评委进行评分：评委独立评估每个组件，并给出加权总分（0–10 分范围，归一化至 0–1）
- 无预设参考答案；评分完全依赖评委判断

## 评估说明

- 默认使用 **test** 划分进行评估（400 个样本）
- 主要指标：**total_score**（各样本归一化后 0–1 总分的平均值）
- 需配置一个能力强的 LLM 评委（例如 GPT-4o、Qwen-Max），通过 `judge_model_args` 设置
- 评委策略应设置为 `JudgeStrategy.LLM`

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mia_bench` |
| **数据集ID** | [lmms-lab/MIA-Bench](https://modelscope.cn/datasets/lmms-lab/MIA-Bench/summary) |
| **论文** | N/A |
| **标签** | `InstructionFollowing`, `MultiModal`, `QA` |
| **指标** | `total_score` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 400 |
| 提示词长度（平均） | 137.81 字符 |
| 提示词长度（最小/最大） | 34 / 327 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 400 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 135x240 - 3264x4928 |
| 格式 | jpeg, mpo, webp |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "9156e81c",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~195.9KB]"
        },
        {
          "text": "Explain the activity taking place in the image using exactly two sentences, including one metaphor."
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "instruction": "Explain the activity taking place in the image using exactly two sentences, including one metaphor.",
    "type": "advanced",
    "num_of_component": 3,
    "components": [
      "Explain the activity taking place in the image",
      "using exactly two sentences",
      "including one metaphor"
    ],
    "component_weight": [
      4,
      3,
      3
    ],
    "component_type": [
      "describe",
      "length_limit",
      "linguistics"
    ]
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
    --datasets mia_bench \
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
    datasets=['mia_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```