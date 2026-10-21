# Seed-TTS-Eval

## 概述

Seed-TTS-Eval 是一个针对零样本文本转语音和声音转换的客观基准测试。它使用来自 Common Voice 和 DiDiSpeech-2 的域外英语和普通话样本，官方评估重点在于可理解性和说话人一致性。

## 任务描述

- **任务类型**: 零样本文本转语音生成
- **输入**: 参考说话人的音频、提示文本和目标文本
- **输出**: 使用参考说话人的目标文本合成的语音音频
- **语言**: 英语和普通话

## 评估说明

- 默认子集: **en** 和 **zh**
- 被评估的 TTS 模型必须返回生成的音频作为 `ContentAudio`，或者返回音频路径、URL 或数据 URI 作为完成文本
- EvalScope 为 HTTP TTS 服务提供了 `eval_type="text2speech"`。
  - Volcengine 提供者: 配置 `model="seed-tts-2.0"`，`api_url="https://openspeech.bytedance.com/api/v3/tts/unidirectional"`，以及 `model_args={"speaker": "..."}`。
  - OpenAI 提供者: 配置 `model="tts-1"`（或 `tts-1-hd`），`api_url="https://api.openai.com/v1"`，以及 `model_args={"provider": "openai", "voice": "nova"}`。
- 默认指标: **audio_wer**，通过 OpenAI 兼容的 `/audio/transcriptions` 端点将生成的音频转录，并计算具有特定语言规范化的 WER/CER 样式错误率
- 通过 `metric_list` 配置 ASR 端点，或设置 `SEED_TTS_EVAL_ASR_API_BASE`、`SEED_TTS_EVAL_ASR_API_KEY`、`SEED_TTS_EVAL_ASR_MODEL` 和 `SEED_TTS_EVAL_ASR_API_PROTOCOL`
- 对于 Volcengine Ark 音频理解模型，设置 `api_protocol="responses"` 并使用支持音频输入的模型，例如 `doubao-seed-2-0-lite-260428`
- 说话人相似性是官方基准测试的一部分，但需要单独的说话人验证后端，默认情况下未启用
- 需要支持 `input_audio` 内容部分的音频输入的 OpenAI 兼容服务器。SGLang 在 v0.5.19 (sgl-project/sglang#33606) 中添加了此支持；早期版本会以 HTTP 400 拒绝音频请求。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `seed_tts_eval` |
| **数据集 ID** | [evalscope/Seed-TTS-Eval](https://modelscope.cn/datasets/evalscope/Seed-TTS-Eval/summary) |
| **论文** | [论文](https://arxiv.org/abs/2406.02430) |
| **标签** | `Audio`, `TextToSpeech` |
| **指标** | `audio_wer` |
| **默认样本数** | 0-shot |
| **评估拆分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,108 |
| 提示词长度（平均） | 243.31 字符 |
| 提示词长度（最小/最大） | 193 / 376 字符 |

**每子集统计:**

| 子集 | 样本数 | 提示词平均 | 提示词最小 | 提示词最大 |
|--------|---------|-------------|------------|------------|
| `en` | 1,088 | 304.3 | 224 | 376 |
| `zh` | 2,020 | 210.46 | 193 | 231 |

**音频统计:**

| 指标 | 值 |
|--------|-------|
| 总音频文件数 | 3,108 |
| 每个样本的音频 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | wav |

## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "ffd452c5",
      "content": [
        {
          "audio": "[BASE64_AUDIO: wav, ~183.0KB]",
          "format": "wav"
        },
        {
          "text": "Use the reference audio and prompt transcript to synthesize the target text in the same speaker voice.\nPrompt transcript: We asked over twenty different people, and they all said it was his.\nTarget text: Get the trust fund to the bank early.\nReturn only the generated audio."
        }
      ]
    }
  ],
  "target": "Get the trust fund to the bank early.",
  "id": 0,
  "group_id": 0,
  "subset_key": "en",
  "metadata": {
    "filename": "common_voice_en_10119832-common_voice_en_10119840",
    "prompt_text": "We asked over twenty different people, and they all said it was his.",
    "text": "Get the trust fund to the bank early.",
    "prompt_audio_path": "prompt-wavs/common_voice_en_10119832.wav",
    "reference_audio_path": "wavs/common_voice_en_10119832-common_voice_en_10119840.wav",
    "language": "en",
    "wer_language": "en"
  }
}
```

## 提示模板

**提示模板:**
```text
Use the reference audio and prompt transcript to synthesize the target text in the same speaker voice.
Prompt transcript: {prompt_text}
Target text: {text}
Return only the generated audio.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets seed_tts_eval \
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
    datasets=['seed_tts_eval'],
    dataset_args={
        'seed_tts_eval': {
            # subset_list: ['en', 'zh']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
