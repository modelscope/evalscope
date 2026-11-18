# LongText-Bench


## 概述

LongText-Bench 评估文本到图像模型是否准确渲染长英文和中文文本。

## 任务描述

- **任务类型**：文本到图像生成
- **输入**：描述包含指定文本的场景的提示词
- **输出**：包含所请求文本的图像
- **领域**：长文本渲染

## 主要特点

- 来自 X-Omni 的 160 个英文和 160 个中文提示词，涵盖海报、网页和幻灯片等 8 种场景。
- 默认加载 ModelScope 镜像；同时也支持官方 Hugging Face 源。

## 评估说明

- 使用视觉 API 转录图像。文本分数（Text Score）为匹配 token 数除以 ground-truth token 数（英文单词或中文字符）。双语总体分数要求每种语言都有评分结果；否则报告将显示可用语言的分数。
- 默认在 DashScope 上使用 qwen3.8-flash：需设置 DASHSCOPE_API_KEY；无需本地 VLM 权重。官方评估器使用 Qwen2.5-VL-7B，因此不同转录模型的结果不可直接比较。
- 评分遵循官方匹配规则；对于其重复图像协议，请使用 repeats=4。文本分数不评估布局或美学效果。API 失败和截断回复将被排除；有效的空转录得分为零。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `longtext_bench` |
| **数据集ID** | [evalscope/LongText-Bench](https://modelscope.cn/datasets/evalscope/LongText-Bench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2507.22058) |
| **标签** | `TextToImage` |
| **指标** | `LongTextScore` |
| **默认 Shots** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 320 |
| 提示词长度（平均值） | 614.67 字符 |
| 提示词长度（最小/最大值） | 181 / 1368 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `en` | 160 | 964.03 | 631 | 1368 |
| `zh` | 160 | 265.32 | 181 | 393 |

## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "a3412209",
      "content": "A lively and bustling urban street scene captured at daytime, featuring a colorful assortment of vibrant storefronts lining the busy sidewalk, with diverse groups of pedestrians leisurely strolling past shop windows, chatting or hurrying alon ... [TRUNCATED 551 chars] ... ed \"CENTRAL PARK - 2 BLOCKS\", guiding visitors with its distinctly readable font. All signage is brightly lit, eye-catching, and clearly contrasts against the detailed backdrop of dynamic city buildings, enhancing the lively urban atmosphere."
    }
  ],
  "target": "FRESH BREWS, EVERY MORNING, EVERY DAY Locally Roasted Since 1998 CENTRAL PARK - 2 BLOCKS",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "category": "sign",
    "length": "short",
    "prompt": "A lively and bustling urban street scene captured at daytime, featuring a colorful assortment of vibrant storefronts lining the busy sidewalk, with diverse groups of pedestrians leisurely strolling past shop windows, chatting or hurrying alon ... [TRUNCATED 551 chars] ... ed \"CENTRAL PARK - 2 BLOCKS\", guiding visitors with its distinctly readable font. All signage is brightly lit, eye-catching, and clearly contrasts against the detailed backdrop of dynamic city buildings, enhancing the lively urban atmosphere.",
    "text": [
      "FRESH BREWS, EVERY MORNING, EVERY DAY",
      "Locally Roasted Since 1998",
      "CENTRAL PARK - 2 BLOCKS"
    ],
    "text_length": 15,
    "prompt_id": 0,
    "tags": [],
    "id": "",
    "image_path": ""
  }
}
```

*注意：部分内容为展示目的已截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets longtext_bench \
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
    datasets=['longtext_bench'],
    dataset_args={
        'longtext_bench': {
            # subset_list: ['en', 'zh']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
