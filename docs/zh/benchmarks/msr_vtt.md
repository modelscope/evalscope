# MSR-VTT


## 概述

MSR-VTT 是一个大规模开放域视频描述（video captioning）基准测试，用于评估视频到文本的生成能力。  
原生适配器按 `video_id` 对记录进行分组，因此同一视频的多条标注行会被合并为一个样本，并包含多个参考描述。

## 任务描述

- **任务类型**：视频描述（Video captioning）
- **输入**：视频片段或 URL
- **输出**：一条简洁的自然语言描述
- **领域**：开放域视频理解与描述

## 评估说明

- 默认数据源：ModelScope 上的 `AI-ModelScope/msr-vtt`，使用 `validation` 划分
- 通过设置 `extra_params.dataset_hub="huggingface"`，仍可使用 Hugging Face 上的 `VLM2Vec/MSR-VTT`
- 主要指标：**CIDEr**
- 其他指标：BLEU-1/2/3/4、METEOR、ROUGE-L
- 设置 `extra_params.video_dir` 可优先使用本地媒体文件而非 URL 元数据

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `msr_vtt` |
| **数据集ID** | [AI-ModelScope/msr-vtt](https://modelscope.cn/datasets/AI-ModelScope/msr-vtt/summary) |
| **论文** | [Paper](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) |
| **标签** | `ImageCaptioning`, `MultiModal` |
| **指标** | `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **默认示例数** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 497 |
| 提示词长度（平均） | 43 字符 |
| 提示词长度（最小/最大） | 43 / 43 字符 |

**视频统计信息：**

| 指标 | 值 |
|--------|-------|
| 视频总数 | 497 |
| 每样本视频数 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp4 |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "36044e4b",
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
    "video_id": "video6513",
    "category": 14
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
| `dataset_hub` | `str` | `modelscope` | 用于加载 MSR-VTT 标注的数据集平台。选项：['huggingface', 'modelscope', 'local'] |
| `eval_split` | `str` | `` | 要加载的数据划分；ModelScope 默认为 validation，Hugging Face 默认为 test。 |
| `dataset_revision` | `str` | `` | 可选的数据集版本；留空则使用平台默认版本。 |
| `video_dir` | `str` | `` | 包含 MSR-VTT 视频文件的本地目录（可选）。 |
| `video_extension` | `str` | `` | 本地视频文件的扩展名覆盖项（可选），例如 "mp4"。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets msr_vtt \
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
    datasets=['msr_vtt'],
    dataset_args={
        'msr_vtt': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```