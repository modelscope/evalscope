# MSVD


## 概述

MSVD 是一个经典的视频描述（video captioning）基准测试，包含大量带有多个人工标注字幕的短视频。
原生适配器将每个视频视为一个评估样本，并使用所有可用字幕作为参考。

## 任务描述

- **任务类型**：视频描述（Video captioning）
- **输入**：视频片段
- **输出**：一条简洁的自然语言字幕
- **领域**：开放域视频理解与描述

## 评估说明

- 默认数据源：ModelScope 上的 `evalscope/MSVD`，使用 `test` 划分
- 通过设置 `extra_params.dataset_hub="huggingface"`，仍可使用 Hugging Face 上的 `VLM2Vec/MSVD`
- 主要指标：**CIDEr**
- 其他指标：BLEU-1/2/3/4、METEOR、ROUGE-L
- 当数据集仅提供视频文件名且需要本地媒体文件时，请设置 `extra_params.video_dir`

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `msvd` |
| **数据集ID** | [evalscope/MSVD](https://modelscope.cn/datasets/evalscope/MSVD/summary) |
| **论文** | [Paper](https://aclanthology.org/P11-1020/) |
| **标签** | `ImageCaptioning`, `MultiModal` |
| **指标** | `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 670 |
| 提示词长度（平均） | 43 字符 |
| 提示词长度（最小/最大） | 43 / 43 字符 |

**视频统计信息：**

| 指标 | 值 |
|--------|-------|
| 视频总数 | 670 |
| 每样本视频数 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp4 |


## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "a4b83275",
      "content": [
        {
          "text": "Describe the video in one concise sentence."
        },
        {
          "video": "fr9H1WLcF1A_256_261.avi",
          "format": "mp4"
        }
      ]
    }
  ],
  "target": "[\"two young men are playing table tennis\", \"men are playing table tennis\", \"two men are playing table tennis\", \"two men are playing a tabletennis\", \"a couple of people are playing a game of ping pong\", \"peoples are playing table tennis\", \"two ... [TRUNCATED 306 chars] ... e boys are playing\", \"people are playing ping pong\", \"18 kids and counting show the newest baby josie\", \"there is some kids and playing to each other\", \"the men played pingpong together\", \"the boys are playing ping pong\", \"2 boys is playing\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "references": [
      "two young men are playing table tennis",
      "men are playing table tennis",
      "two men are playing table tennis",
      "two men are playing a tabletennis",
      "a couple of people are playing a game of ping pong",
      "peoples are playing table tennis",
      "two men are playing pingpong",
      "two guys play table tennis",
      "two people are playing ping pong",
      "two boys are playing ping pong",
      "... [TRUNCATED 12 more items] ..."
    ],
    "subset": "default",
    "dataset_id": "evalscope/MSVD",
    "dataset_hub": "modelscope",
    "video": "fr9H1WLcF1A_256_261.avi",
    "video_id": "fr9H1WLcF1A_256_261",
    "source": "MSVD"
  }
}
```

## 提示模板

**提示模板：**
```text
Describe the video in one concise sentence.
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `dataset_hub` | `str` | `modelscope` | 用于加载 MSVD 标注的数据集平台。选项：['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | 要加载的数据划分；默认为 test。 |
| `dataset_revision` | `str` | `` | 可选的数据集版本；留空则使用平台默认版本。 |
| `video_dir` | `str` | `` | 包含 MSVD 视频文件的本地目录（可选）。 |
| `video_extension` | `str` | `` | 本地视频文件的扩展名覆盖项（可选），例如 "mp4"。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets msvd \
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
    datasets=['msvd'],
    dataset_args={
        'msvd': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```