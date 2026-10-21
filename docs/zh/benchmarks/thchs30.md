# THCHS-30

## 概述

THCHS-30 是一个带有音素级时间对齐的普通话朗读语音语料库。此EvalScope基准测试使用随数据集发布的对齐IPA音素序列来评估从语音中识别音素的能力。

## 任务描述

- **任务类型**: 音素识别
- **输入**: 16 kHz的普通话录音
- **输出**: 带有声调符号的空格分隔的IPA音素序列
- **领域**: 普通话朗读语音

## 主要特点

- 13,388条语音，分为10,000条训练样本、893条验证样本和2,495条测试样本
- 包括`[SIL]`静音间隔在内的音素级起始和结束时间戳
- 保留原始汉字转录、说话人标识符、持续时间和分割标签作为样本元数据
- 测试评估使用发布时的IPA音素库存和声调注释

## 评估说明

- 默认配置仅评估`test`分割；不使用少量示例
- 主要指标：语料库级别的音素错误率（PER），计算为总音素标记编辑距离除以总参考音素数；报告计数保持为话语计数
- 模型必须仅输出由空格分隔的IPA音素标记；时间戳是用于下游分析的元数据，不是模型的目标
- 音频通过EvalScope的标准音频消息格式以WAV数据形式传递
- 需要支持`input_audio`内容部分的OpenAI兼容服务器，用于音频输入

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `thchs30` |
| **数据集ID** | [evalscope/THCHS-30](https://modelscope.cn/datasets/evalscope/THCHS-30/summary) |
| **论文** | [论文](https://arxiv.org/abs/1512.01882) |
| **标签** | `Audio`, `Chinese`, `SpeechRecognition` |
| **指标** | `per` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |
| **聚合方式** | `weighted_mean` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,495 |
| 提示词长度（平均） | 88个字符 |
| 提示词长度（最小/最大） | 88 / 88个字符 |

**音频统计:**

| 指标 | 值 |
|--------|-------|
| 总音频文件数 | 2,495 |
| 每个样本的音频 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | wav |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "307a05d8",
      "content": [
        {
          "text": "Transcribe the audio as IPA phone tokens. Output only tokens separated by single spaces."
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~284.9KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "[SIL] t a˥ ŋ ɻ a˧˥ː n [SIL] tʰ˘ a˥˘ m˘ ə n˘ p u˥˩˘ ɕ j a˥˩˘ ŋ ɤ˥˩ː y˧˥ n a˥˩ jˑ a˥˩˘ ŋ˘ tʰ w˘ ə˥˘ n ɕ˘ j a˥˩ ʂː ɻ̩˧˥ kʰ w ai̯˥˩ˑ [SIL] t a˥ ŋ ts˘ wˑ o˥˩ˑ jː a˥ tsʰ a˥ ŋ ʂː ɻ̩˧˥ˑ [SIL] ɚ˧˥˘ ʂ ɻ̩˥˩ mˑ w o˧˥ ʈʂʰ˘ ə˧˥ ŋ f ə˧˩˧ nː i˧˩˧ xː ou̯˥˩ tsʰ˘ ai̯˧˥˘ fˑ u˧˥ˑ j ʊ˥˩ ŋ [SIL]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "utt_id": "D11_779",
    "text": "当然 他们 不 像 鳄鱼 那样 吞下 石块 当做 压 舱 石 而是 磨成 粉 以后 才 服用",
    "phones": [
      "[SIL]",
      "t",
      "a˥",
      "ŋ",
      "ɻ",
      "a˧˥ː",
      "n",
      "[SIL]",
      "tʰ˘",
      "a˥˘",
      "... [TRUNCATED 66 more items] ..."
    ],
    "phone_starts": [
      0.0,
      1.55,
      1.6,
      1.69,
      1.75,
      1.81,
      1.96,
      2.03,
      2.33,
      2.39,
      "... [TRUNCATED 66 more items] ..."
    ],
    "phone_ends": [
      1.55,
      1.6,
      1.69,
      1.75,
      1.81,
      1.96,
      2.03,
      2.33,
      2.39,
      2.43,
      "... [TRUNCATED 66 more items] ..."
    ],
    "language": "cmn",
    "speaker_id": "D11",
    "duration": 9.114,
    "split": "test"
  }
}
```

*注意: 为了显示，部分内容被截断。*

## 提示模板

**提示模板:**
```text
Transcribe the audio as IPA phone tokens. Output only tokens separated by single spaces.
```

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets thchs30 \
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
    datasets=['thchs30'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
