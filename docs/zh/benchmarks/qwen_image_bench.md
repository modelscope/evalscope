# Qwen-Image-Bench


## 概述

Qwen-Image-Bench 使用官方 Q-Judger 评估文本到图像模型在专业创意任务上的表现。

## 任务描述

- **任务类型**：文本到图像生成与评估
- **输入**：中文或英文创意提示词
- **输出**：根据适用评分标准进行评估的图像
- **领域**：图像质量、美学、对齐度、真实世界保真度及创意生成

## 主要特点

- 包含 1,000 条双语提示词，涵盖归入五个维度的 56 个细分项。
- 支持通过本地 JSONL 清单进行图像生成以及现有图像的评估。

## 评估说明

- 将 `judge.models` 设置为提供官方 `Qwen/Qwen-Image-Bench` 权重的 OpenAI 兼容视觉端点。启用思考模式，并将推理过程与最终 JSON 分离。官方设置下使用温度 0、top-k 1、top-p 1、重复惩罚 1.05、种子 42 以及 4,096 token 的生成限制。
- 将 `extra_params.language` 设置为 `cn`（默认）或 `en`。每次运行报告所选语言的分数。
- 对于现有图像，请使用包含 `ID`、`dims_en`、`prompt`（或选定的官方 `prompt_cn`/`prompt_en` 字段）以及绝对路径 `image_path` 的本地 JSONL 文件。设置 `model_task='image_generation'` 并保持 `model` 未设置以对这些图像进行评分。
- 等级 0、1 和 2 分别映射为 0、60 和 100 分。不适用的细分项和失败的请求将从相应的平均值中排除。每张图像先在其得分维度上取平均，然后再对图像进行平均。
- 总体分数、各维度及各子维度均报告其各自的有效图像数量。

此配置针对英文提示词对现有图像进行评分。

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model_id='my-image-model',
    model_task='image_generation',
    datasets=['qwen_image_bench'],
    dataset_args={'qwen_image_bench': {
        'dataset_id': '/absolute/path/to/manifest.jsonl',
        'extra_params': {'language': 'en'},
    }},
    judge={'models': [{
        'model_id': 'Qwen/Qwen-Image-Bench',
        'api_url': 'http://127.0.0.1:8000/v1',
        'generation_config': {
            'temperature': 0,
            'top_p': 1,
            'seed': 42,
            'max_tokens': 4096,
            'extra_body': {
                'top_k': 1,
                'repetition_penalty': 1.05,
                'chat_template_kwargs': {'enable_thinking': True},
            },
        },
    }]},
))
```


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `qwen_image_bench` |
| **数据集ID** | [Qwen/Qwen-Image-Bench](https://modelscope.cn/datasets/Qwen/Qwen-Image-Bench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2605.28091) |
| **标签** | `TextToImage` |
| **指标** | `qwen_image_bench_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（均值） | 83.05 字符 |
| 提示词长度（最小/最大） | 25 / 317 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "b8c6bf0d",
      "content": "一只拆开的机械手表放在黑色工作台上，制表师用镊子夹起齿轮，特写镊子夹起齿轮，银色与暗灰色调，零部件结构精密，金属边缘细节清晰。"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": 1,
    "prompt": "一只拆开的机械手表放在黑色工作台上，制表师用镊子夹起齿轮，特写镊子夹起齿轮，银色与暗灰色调，零部件结构精密，金属边缘细节清晰。",
    "dimensions": [
      "Quality",
      "Aesthetics",
      "Alignment"
    ],
    "image_path": "",
    "category": "",
    "tags": []
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `language` | `str` | `cn` | 生成提示词的语言（cn 或 en）。选项：['cn', 'en'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets qwen_image_bench \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['qwen_image_bench'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'qwen_image_bench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
