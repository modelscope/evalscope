# Video-MME-v2


## 概述

Video-MME-v2 是一个公开的综合性视频理解基准测试。它包含 800 个视频、3,200 个多选问答样本，以及带有时间戳的词级字幕。该基准测试的原生适配器使用共享的 `DatasetHub` 抽象来加载标注数据并可选地下载媒体归档文件，因此其复用了与 MVBench 相同的可重用视频基准测试路径。

## 任务描述

- **任务类型**：视频多选问答（MCQ）
- **输入**：视频 URL 或归档的 MP4 文件 + 问题 + 答案选项
- **输出**：单个正确答案字母
- **子集**：`all`、`level_1`、`level_2`、`level_3`、`logic`、`relevance`

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率（Accuracy）**
- 默认视频源为公开的 `url` 字段，用于轻量级冒烟测试
- 将 `extra_params.video_source` 设置为 `archive` 以下载并使用官方 MP4 归档文件
- 将 `extra_params.use_subtitles` 设置为 `true` 以在提示中包含词级字幕

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `videomme_v2` |
| **数据集ID** | [MME-Benchmarks/Video-MME-v2](https://modelscope.cn/datasets/MME-Benchmarks/Video-MME-v2/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2604.05015) |
| **标签** | `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,200 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `all` | 3,200 | N/A | N/A | N/A |
| `level_1` | 686 | N/A | N/A | N/A |
| `level_2` | 834 | N/A | N/A | N/A |
| `level_3` | 837 | N/A | N/A | N/A |
| `logic` | 1,124 | N/A | N/A | N/A |
| `relevance` | 2,076 | N/A | N/A | N/A |

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `dataset_id` | `str` | `MME-Benchmarks/Video-MME-v2` | Video-MME-v2 的数据集仓库 ID 或本地数据集根目录。 |
| `dataset_hub` | `str` | `modelscope` | 用于加载标注、字幕和可选视频归档的 dataset hub。选项：['huggingface', 'modelscope', 'local'] |
| `dataset_revision` | `str` | `` | 可选的数据集版本；留空则使用 hub 默认版本。 |
| `video_source` | `str` | `url` | 使用公开 URL 字段进行轻量级测试，或使用官方归档的 MP4 文件。选项：['url', 'archive'] |
| `use_subtitles` | `bool` | `False` | 在提示中包含 Video-MME-v2 的字幕文本。 |
| `subtitle_word_limit` | `int` | `512` | 启用字幕时，每个样本最多包含的字幕词数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets videomme_v2 \
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
    datasets=['videomme_v2'],
    dataset_args={
        'videomme_v2': {
            # subset_list: ['all', 'level_1', 'level_2']  # 可选，用于评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```