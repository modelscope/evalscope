# MSVD


## 概览

MSVD 是一个经典的视频描述基准，包含带有大量人工描述标注的短网络视频。
原生适配器会将每个视频视为一个评测样本，并使用所有可用描述作为参考答案。

## 任务描述

- **任务类型**：视频描述
- **输入**：视频片段
- **输出**：一句简洁的自然语言描述
- **领域**：开放域视频理解与描述

## 评测说明

- 默认数据源：ModelScope 上的 `evalscope/MSVD`，`test` 划分
- 也可以通过设置 `extra_params.dataset_hub="huggingface"` 使用 Hugging Face 上的 `VLM2Vec/MSVD`
- 主要指标：**CIDEr**
- 其他指标：BLEU-1/2/3/4、METEOR、ROUGE-L
- 当数据集只提供视频文件名且需要本地媒体文件时，设置 `extra_params.video_dir`


## 属性

| 属性 | 值 |
|----------|-------|
| **基准名称** | `msvd` |
| **数据集 ID** | [evalscope/MSVD](https://modelscope.cn/datasets/evalscope/MSVD/summary) |
| **论文** | [论文](https://aclanthology.org/P11-1020/) |
| **标签** | `ImageCaptioning`, `MultiModal` |
| **指标** | `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **默认样本数** | 0-shot |
| **评测划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 样本总数 | 670 |
| Prompt 平均长度 | 43 个字符 |
| Prompt 最小/最大长度 | 43 / 43 个字符 |

**视频统计：**

| 指标 | 值 |
|--------|-------|
| 视频总数 | 670 |
| 每个样本的视频数量 | min: 1, max: 1, mean: 1 |
| 格式 | mp4 |


## 样例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "b91ac25b",
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
    "dataset_id": "VLM2Vec/MSVD",
    "dataset_hub": "huggingface",
    "video": "fr9H1WLcF1A_256_261.avi",
    "start": null,
    "end": null,
    "fps": null,
    "media_resolved": false,
    "id": null,
    "video_id": "fr9H1WLcF1A_256_261",
    "image_id": null,
    "question_id": null,
    "source": "MSVD",
    "category": null
  }
}
```

## Prompt 模板

**Prompt 模板：**
```text
Describe the video in one concise sentence.
```

## 额外参数

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `dataset_hub` | `str` | `modelscope` | 用于加载 MSVD 标注的数据集平台。可选值：['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | 要加载的源数据划分；默认使用 test。 |
| `dataset_revision` | `str` | `` | 可选的数据集版本；留空时使用平台默认版本。 |
| `video_dir` | `str` | `` | 可选的本地目录，包含 MSVD 视频文件。 |
| `video_extension` | `str` | `` | 可选的本地视频扩展名覆盖值，例如 "mp4"。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets msvd \
    --limit 10  # 正式评测时移除此行
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
    limit=10,  # 正式评测时移除此行
)

run_task(task_cfg=task_cfg)
```
