# WenetSpeech

## 概述

WenetSpeech 是一个大规模的中文语音语料库，包含超过10,000小时的多领域转录音频数据，专为语音识别研究设计。

## 任务描述

- **任务类型**: 自动语音识别 (ASR)
- **输入**: 包含中文语音的音频记录
- **输出**: 中文转录文本
- **领域**: 多领域（互联网、会议）

## 主要特点

- 大规模中文语音语料库（10,000+小时）
- 多领域覆盖：互联网内容、会议
- 高质量转录
- 适合评估中文ASR系统
- 支持中英文混合文本评估

## 评估说明

- 默认配置使用 **test_meeting** 分割
- 按领域划分的子集：**dev**（开发）、**test_meeting**（会议领域）
- 主要指标：**MER**（混合错误率）
- MER将中文字符单独分词，英文单词作为整体分词
- 提示：“请听音频并转录你听到的内容”
- 需要支持 `input_audio` 内容部分的OpenAI兼容服务器

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `wenet_speech` |
| **数据集ID** | [lmms-lab/WenetSpeech](https://modelscope.cn/datasets/lmms-lab/WenetSpeech/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `SpeechRecognition` |
| **指标** | `mer` |
| **默认样本数** | 0-shot |
| **评估分割** | `test_meeting` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 22,195 |
| 提示词长度（平均） | 161 字符 |
| 提示词长度（最小/最大） | 161 / 161 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `dev` | 13,825 | 161 | 161 | 161 |
| `test_meeting` | 8,370 | 161 | 161 | 161 |

**音频统计：**

| 指标 | 值 |
|--------|-------|
| 总音频文件数 | 22,195 |
| 每个样本的音频数量 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | wav |

## 样例示例

**子集**: `dev`

```json
{
  "input": [
    {
      "id": "c30c80b4",
      "content": [
        {
          "text": "Please listen to the audio and transcribe what you hear. Please only provide the transcription without any additional commentary. Do not include any punctuation."
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~175.3KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "text": "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯"
  }
}
```

## 提示模板

**提示模板:**
```text
Please listen to the audio and transcribe what you hear. Please only provide the transcription without any additional commentary. Do not include any punctuation.
```

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets wenet_speech \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['wenet_speech'],
    dataset_args={
        'wenet_speech': {
            # subset_list: ['dev', 'test_meeting']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
