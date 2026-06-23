# WenetSpeech


## 概述

WenetSpeech 是一个大规模中文语音语料库，包含超过 10,000 小时的多领域转录音频数据，专为语音识别研究而设计。

## 任务描述

- **任务类型**：自动语音识别（ASR）
- **输入**：包含中文语音的音频录音
- **输出**：中文转录文本
- **领域**：多领域（互联网、会议）

## 主要特性

- 大规模中文语音语料库（10,000+ 小时）
- 覆盖多领域：互联网内容、会议
- 高质量转录
- 适用于评估中文 ASR 系统
- 支持中英混合文本评估

## 评估说明

- 默认配置使用 **test_meeting** 子集
- 按领域划分的子集：**dev**（开发集）、**test_meeting**（会议领域）
- 主要指标：**MER**（Mixed Error Rate，混合错误率）
- MER 将中文字符逐字切分，英文单词作为整体切分
- 提示词："Please listen to the audio and transcribe what you hear"

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `wenet_speech` |
| **数据集ID** | [lmms-lab/WenetSpeech](https://modelscope.cn/datasets/lmms-lab/WenetSpeech/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `SpeechRecognition` |
| **指标** | `mer` |
| **默认样本数** | 0-shot |
| **评估子集** | `test_meeting` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
Please listen to the audio and transcribe what you hear. Please only provide the transcription without any additional commentary. Do not include any punctuation.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets wenet_speech \
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
    datasets=['wenet_speech'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```