# LibriSpeech

## 概述

LibriSpeech 是一个大规模的语料库，包含大约 1000 小时的有声读物中的英语朗读语音。它是评估自动语音识别（ASR）系统最常用的基准之一。

## 任务描述

- **任务类型**: 自动语音识别 (ASR)
- **输入**: 有声读物中的英语朗读语音录音
- **输出**: 转录文本
- **语言**: 英语

## 主要特点

- 1000 小时高质量的朗读语音
- 来源于 LibriVox 有声读物（公共领域）
- 清晰和“其他”测试集用于不同难度
- ASR 研究中广泛使用的基线
- 标准化的评估协议

## 评估说明

- 支持的评估分割: **test_clean** 和 **test_other**
- 主要指标: **词错误率 (WER)**
- 评估过程中应用了文本规范化
- 提示: "请识别语音并仅输出识别内容"
- 元数据包括音频 ID 和持续时间信息
- 需要支持 `input_audio` 内容部分的 OpenAI 兼容服务器。SGLang 在 v0.5.19 (sgl-project/sglang#33606) 中添加了此支持；早期版本会以 HTTP 400 拒绝音频请求。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `librispeech` |
| **数据集ID** | [lmms-lab/Librispeech-concat](https://modelscope.cn/datasets/lmms-lab/Librispeech-concat/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `SpeechRecognition` |
| **指标** | `wer` |
| **默认样本数** | 0-shot |
| **评估分割** | `N/A` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 177 |
| 提示词长度（平均值） | 67 字符 |
| 提示词长度（最小/最大值） | 67 / 67 字符 |

**子集统计数据:**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `test_clean` | 87 | 67 | 67 | 67 |
| `test_other` | 90 | 67 | 67 | 67 |

**音频统计数据:**

| 指标 | 值 |
|--------|-------|
| 总音频文件数 | 177 |
| 每个样本的音频数量 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | wav |

## 样例示例

**子集**: `test_clean`

```json
{
  "input": [
    {
      "id": "732c90a4",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~22.7MB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "Eleven o'clock had struck it was a fine clear night they were the only persons on the road and they sauntered leisurely along to avoid paying the price of fatigue for the recreation provided for the toledans in their valley or on the banks of ... [TRUNCATED 7380 chars] ...  less surprised than they and the better to assure himself of so wonderful a fact he begged leocadia to give him some token which should make perfectly clear to him that which indeed he did not doubt since it was authenticated by his parents.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "audio_id": "5639-40744",
    "audio_duration": 496.6899719238281
  }
}
```

*注：部分内容为显示而被截断。*

## 提示模板

**提示模板:**
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
    --datasets librispeech \
    --limit 10  # Remove this line for formal evaluation
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['librispeech'],
    dataset_args={
        'librispeech': {
            # subset_list: ['test_clean', 'test_other']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
