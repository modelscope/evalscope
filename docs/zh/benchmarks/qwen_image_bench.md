# Qwen-Image-Bench


## 概述

Qwen-Image-Bench 用于衡量文本到图像模型在处理创意任务时的表现。

## 任务描述

- **任务类型**：文本到图像生成
- **输入**：中文或英文提示词
- **输出**：生成的图像
- **领域**：图像质量、美学、提示词对齐度、真实世界保真度和创造力

## 主要特点

- 包含 1,000 条双语提示词。将 `extra_params.language` 设置为 `cn`（默认）或 `en`。
- 使用本地 JSONL 文件对现有图像进行评分（需包含 `ID`、`prompt`、`dims_en` 以及绝对路径的 `image_path`）。
  设置 `model_task='image_generation'` 并省略 `model` 参数。

## 评估说明

- 配置 `judge.models`，指向提供官方 Q-Judger (`Qwen/Qwen-Image-Bench`) 服务的 OpenAI 兼容端点。请参阅[官方推理设置](https://github.com/QwenLM/Qwen-Image-Bench#inference-parameters)。
- 分数范围为 0 到 100，分数越高越好。报告将显示总体得分及细分得分，并列出有效图像的数量。


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
| 提示词长度（平均值） | 83.05 字符 |
| 提示词长度（最小/最大值） | 25 / 317 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "58f90ec0",
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
| `language` | `str` | `cn` | 生成提示词的语言（cn 或 en）。可选值：['cn', 'en'] |

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
