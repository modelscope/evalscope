# MSR-VTT


## 概览

MSR-VTT 是一个大规模开放域视频描述基准，用于评测视频到文本生成能力。
原生适配器会按 `video_id` 对记录分组，因此同一视频的多条标注记录会合并为一个样本，并保留多条参考描述。

## 任务描述

- **任务类型**：视频描述
- **输入**：视频片段或 URL
- **输出**：一句简洁的自然语言描述
- **领域**：开放域视频理解与描述

## 评测说明

- 默认数据源：ModelScope 上的 `AI-ModelScope/msr-vtt`，`validation` 划分
- 也可以通过设置 `extra_params.dataset_hub="huggingface"` 使用 Hugging Face 上的 `VLM2Vec/MSR-VTT`
- 主要指标：**CIDEr**
- 其他指标：BLEU-1/2/3/4、METEOR、ROUGE-L
- 设置 `extra_params.video_dir` 后，会优先使用本地媒体文件，而不是 URL 元数据


## 属性

| 属性 | 值 |
|----------|-------|
| **基准名称** | `msr_vtt` |
| **数据集 ID** | [AI-ModelScope/msr-vtt](https://modelscope.cn/datasets/AI-ModelScope/msr-vtt/summary) |
| **论文** | [论文](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) |
| **标签** | `ImageCaptioning`, `MultiModal` |
| **指标** | `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **默认样本数** | 0-shot |
| **评测划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 样本总数 | 497 |
| Prompt 平均长度 | 43 个字符 |
| Prompt 最小/最大长度 | 43 / 43 个字符 |

**视频统计：**

| 指标 | 值 |
|--------|-------|
| 视频总数 | 497 |
| 每个样本的视频数量 | min: 1, max: 1, mean: 1 |
| 格式 | mp4 |


## 样例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "40a7606c",
      "content": [
        {
          "text": "Describe the video in one concise sentence."
        },
        {
          "video": "https://www.youtube.com/watch?v=A9pM9iOuAzM",
          "format": "mp4",
          "start": 116.03,
          "end": 126.21
        }
      ]
    }
  ],
  "target": "[\"a family is having coversation\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "references": [
      "a family is having coversation"
    ],
    "subset": "default",
    "dataset_id": "AI-ModelScope/msr-vtt",
    "dataset_hub": "modelscope",
    "video": "https://www.youtube.com/watch?v=A9pM9iOuAzM",
    "start": 116.03,
    "end": 126.21,
    "fps": null,
    "media_resolved": true,
    "id": 6513,
    "video_id": "video6513",
    "image_id": null,
    "question_id": null,
    "source": null,
    "category": 14
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
| `dataset_hub` | `str` | `modelscope` | 用于加载 MSR-VTT 标注的数据集平台。可选值：['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | 要加载的源数据划分；ModelScope 默认使用 validation，Hugging Face 默认使用 test。 |
| `dataset_revision` | `str` | `` | 可选的数据集版本；留空时使用平台默认版本。 |
| `video_dir` | `str` | `` | 可选的本地目录，包含 MSR-VTT 视频文件。 |
| `video_extension` | `str` | `` | 可选的本地视频扩展名覆盖值，例如 "mp4"。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets msr_vtt \
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
    datasets=['msr_vtt'],
    dataset_args={
        'msr_vtt': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评测时移除此行
)

run_task(task_cfg=task_cfg)
```
