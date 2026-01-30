# FLEURS


## 概述

FLEURS（Few-shot Learning Evaluation of Universal Representations of Speech）是一个覆盖102种语言的大规模多语言基准测试，用于评估自动语音识别（ASR）、口语理解以及语音翻译任务。

## 任务描述

- **任务类型**：自动语音识别（ASR）
- **输入**：包含多种语言语音的音频录音
- **输出**：对应语言的转录文本
- **语言**：涵盖102种语言，包括简体中文、粤语、英语等

## 主要特点

- 覆盖102种语言的大规模多语言数据
- 源自FLoRes-101机器翻译基准测试
- 包含多样化的语系和文字系统
- 高质量的人工录音与转录
- 元数据包含性别、语系分组和说话人信息

## 评估说明

- 默认配置使用 **test** 数据划分
- 主要指标：**词错误率（Word Error Rate, WER）**
- 默认子集：`cmn_hans_cn`（普通话）、`en_us`（英语）、`yue_hant_hk`（粤语）
- 评估过程中应用语言特定的文本归一化
- 提示词：**"Please recognize the speech and only output the recognized content"**

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `fleurs` |
| **数据集ID** | [lmms-lab/fleurs](https://modelscope.cn/datasets/lmms-lab/fleurs/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `MultiLingual`, `SpeechRecognition` |
| **指标** | `wer` |
| **默认样本数（Shots）** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,411 |
| 提示词长度（平均） | 67 字符 |
| 提示词长度（最小/最大） | 67 / 67 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `cmn_hans_cn` | 945 | 67 | 67 | 67 |
| `en_us` | 647 | 67 | 67 | 67 |
| `yue_hant_hk` | 819 | 67 | 67 | 67 |

**音频统计数据：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 2,411 |
| 每样本音频数量 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | wav |


## 样例示例

**子集**: `cmn_hans_cn`

```json
{
  "input": [
    {
      "id": "daf508c3",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~648.8KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束 也 是 新 篇 章 的 开 始",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": 1906,
    "num_samples": 166080,
    "raw_transcription": "“这并不是告别。这是一个篇章的结束，也是新篇章的开始。”",
    "language": "Mandarin Chinese",
    "gender": 0,
    "lang_id": "cmn_hans",
    "lang_group_id": 6
  }
}
```

## 提示模板

**提示模板：**
```text
Please recognize the speech and only output the recognized content:
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets fleurs \
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
    datasets=['fleurs'],
    dataset_args={
        'fleurs': {
            # subset_list: ['cmn_hans_cn', 'en_us', 'yue_hant_hk']  # 可选，用于指定评估子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```