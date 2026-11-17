# LongText-Bench


## 概述

LongText-Bench (X-Omni, arXiv:2507.22058) 通过 160 个英文提示词和 160 个中文提示词，在 8 种富含文本的场景（招牌、带标签物体、印刷材料、网页、幻灯片、海报、字幕、对话）中评估长文本渲染能力。

## 任务描述

- **任务类型**: 文本到图像生成评估（文本渲染）
- **输入**: 描述富含文本场景的文本提示词
- **输出**: 生成的图像由视觉语言模型 (VLM) 转录，并与真实文本进行匹配
- **领域**: 面向文生图模型的长文本渲染

## 主要特点

- 来自官方 X-Omni 发布的 8 种富含文本场景类别中的 160 个英文 + 160 个中文提示词
- Text Score 将 VLM 转录结果与真实片段进行比较，英文采用单词级匹配，中文采用字符级匹配，并按官方评估器的方式进行语料库聚合
- 根据真实文本逐样本检测语言，因此单次运行即可覆盖两种语言

## 评估说明

- 真实值来源于数据集的 ``text`` 字段（渲染片段的列表，像官方评估器那样用空格连接），作为样本目标传递给评分器
- Text Score 是基于 Counter 交集的匹配比率：英文为单词级，中文为字符级（去除空白符）。每种语言的分数为总匹配数 / 总真实值 token 数；总体分数取英文和中文分数的平均值，这与官方排行榜一致。仅评估一种语言时返回该语言的分数。
- 默认加载 ModelScope 镜像，`en` 和 `zh` 配置中各包含 160 个测试示例。原始数据：[X-Omni/LongText-Bench](https://huggingface.co/datasets/X-Omni/LongText-Bench)。
- 转录使用兼容 OpenAI 的视觉 API，默认在 DashScope 上使用 qwen3.8-flash 并禁用思考模式。需设置 DASHSCOPE_API_KEY；无需本地 VLM 权重或加速器。指标参数 `model`, `api_url`, `api_key` 和 `generation_config` 可独立于图像生成模型配置不同的转录服务。
- 匹配和语言聚合遵循官方评估器。官方转录模型为 Qwen2.5-VL-7B；使用 qwen3.8-flash 或其他 VLM 可能会改变分数，因此这些结果必须标明转录模型，且不能直接与官方结果比较。
- 转录失败和 API token 限制截断会排除受影响的指标，而不是赋零分；有效的空转录得分为零。
- 设置 `model_task='image_generation'` 和 `eval_type='text2image'`。对于官方的重复图像协议，请使用 `repeats=4`；`limit` 仅用于冒烟测试运行。


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
| 提示词长度 (均值) | 614.67 chars |
| 提示词长度 (最小/最大) | 181 / 1368 chars |

**子集统计:**

| 子集 | 样本数 | 提示词均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `en` | 160 | 964.03 | 631 | 1368 |
| `zh` | 160 | 265.32 | 181 | 393 |

## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "e0b33751",
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
    "prompt_id": 0
  }
}
```

*注意：部分内容为显示目的已截断。*

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
