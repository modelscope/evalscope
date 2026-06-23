# CommonVoice15


## 概述

Common Voice 15 是由 Mozilla 收集的大规模多语言语音语料库，涵盖 114 种语言，包含来自全球志愿者贡献的数千小时经过验证的语音数据。

## 任务描述

- **任务类型**：自动语音识别（ASR）
- **输入**：包含多种语言语音的音频录音
- **输出**：对应语言的转录文本
- **语言**：114 种语言，包括英语、中文普通话、法语等

## 主要特点

- 由社区贡献并经社区验证的语音录音
- 多样化的说话人人口统计特征（年龄、性别、口音）
- 多种语言，每种语言的数据量各不相同
- 采用 CC-0 许可证，允许开放研究和商业用途
- 高质量的转录文本，经多位听众验证

## 评估说明

- 默认配置使用 **test** 数据划分
- 主要评估指标：**词错误率（WER）**
- 默认子集：`en`（英语）、`zh-CN`（中文普通话）、`fr`（法语）
- 评估过程中应用语言特定的文本归一化
- 提示词："Please recognize the speech and only output the recognized content"

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `common_voice_15` |
| **数据集ID** | [lmms-lab/common_voice_15](https://modelscope.cn/datasets/lmms-lab/common_voice_15/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `MultiLingual`, `SpeechRecognition` |
| **指标** | `wer` |
| **默认样本数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 43,143 |
| 提示词长度（平均） | 67 字符 |
| 提示词长度（最小/最大） | 67 / 67 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `en` | 16,386 | 67 | 67 | 67 |
| `zh-CN` | 10,625 | 67 | 67 | 67 |
| `fr` | 16,132 | 67 | 67 | 67 |

**音频统计数据：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 43,143 |
| 每样本音频数量 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp3 |


## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "88959854",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: mp3, ~37.0KB]",
          "format": "mp3"
        }
      ]
    }
  ],
  "target": "Joe Keaton disapproved of films, and Buster also had reservations about the medium.",
  "id": 0,
  "group_id": 0,
  "subset_key": "en",
  "metadata": {
    "locale": "en",
    "path": "/home/tiger/.cache/huggingface/datasets/downloads/extracted/f54628fae82dd952031cdea3ec9c3d600c11d606e00cb8b3fd1b6ad500d7eb23/en_test_0/common_voice_en_27710027.mp3",
    "lang_id": "en"
  }
}
```

## 提示模板

**提示模板：**
```text
Please recognize the speech and only output the recognized content:
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets common_voice_15 \
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
    datasets=['common_voice_15'],
    dataset_args={
        'common_voice_15': {
            # subset_list: ['en', 'zh-CN', 'fr']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```