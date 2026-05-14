# Video-MME-v2

## 概述

Video-MME-v2 是一个公开视频理解评测集，用于评估多模态模型的综合视频理解能力。它包含 800 个视频、3,200 条多选问答样本，以及带时间戳的词级字幕文件。

这个原生适配器刻意复用了与 MVBench 相同的视频 benchmark 路径：通过 `DatasetHub` 加载标注，通过同一个 hub 抽象解析可选媒体归档，并使用 `ContentVideo` 与 OpenAI 兼容的 `video_url` 内容路径构造样本。

## 任务描述

- **任务类型**：视频多选问答
- **输入**：公开视频 URL 或官方归档 MP4 + 问题 + 候选答案
- **输出**：单个答案字母
- **默认子集**：`all`
- **完整评测**：800 个视频上的 3,200 条问答样本

## 主要特性

- 使用公开评测集中的真实视频输入
- 复用 `VisionLanguageAdapter`、`MultiChoiceAdapter`、`DatasetHub` 和 `ContentVideo`
- 支持轻量的公开视频 URL 模式，便于小批量 smoke test
- 通过 `extra_params.video_source = "archive"` 支持官方 MP4 归档模式
- 可通过 `extra_params.use_subtitles = true` 将词级字幕加入 prompt
- 支持子集：`all`、`level_1`、`level_2`、`level_3`、`logic`、`relevance`

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率（Accuracy）**
- 公开 ModelScope 数据集将官方视频存放在 40 个大型 zip 归档中；URL 模式可避免小批量测试时下载多 GB 视频包
- 归档模式更利于复现，但即使只评测一个样本也可能下载数 GB

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `videomme_v2` |
| **数据集ID** | [MME-Benchmarks/Video-MME-v2](https://modelscope.cn/datasets/MME-Benchmarks/Video-MME-v2) |
| **论文** | [Video-MME-v2](https://arxiv.org/abs/2604.05015) |
| **标签** | `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估数据划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,200 |
| 视频数量 | 800 |
| 每个视频的问题数 | 4 |

## 样例示例

```json
{
  "video_id": "001",
  "url": "https://www.youtube.com/watch?v=AYSYelOQtQI",
  "question_id": "001-1",
  "question": "What is the ethnicity of the protagonist's mother?",
  "options": "A. Malaysian.\nB. British.\nC. Singaporean.\nD. German.\nE. Canadian.\nF. Chinese.\nG. American.\nH. Cannot be determined.",
  "answer": "F"
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets videomme_v2 \
    --dataset-args '{"videomme_v2": {"subset_list": ["all"], "extra_params": {"dataset_hub": "modelscope", "video_source": "url", "use_subtitles": true, "subtitle_word_limit": 512}}}' \
    --limit 2
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['videomme_v2'],
    dataset_args={
        'videomme_v2': {
            'subset_list': ['all'],
            'extra_params': {
                'dataset_hub': 'modelscope',
                'video_source': 'url',
                'use_subtitles': True,
                'subtitle_word_limit': 512,
            },
        }
    },
    limit=2,
)

run_task(task_cfg=task_cfg)
```
