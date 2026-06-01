# Seed-TTS-Eval


## 概述

Seed-TTS-Eval 是一个用于零样本文本到语音（TTS）和语音转换评估的客观基准测试。它使用来自 Common Voice 和 DiDiSpeech-2 的域外英文和中文样本，官方评估重点关注语音可懂度和说话人一致性。

## 任务描述

- **任务类型**：零样本文本到语音生成
- **输入**：参考说话人音频、提示文本转录和目标文本
- **输出**：使用参考说话人声音合成的目标文本语音音频
- **语言**：英文和中文

## 评估说明

- 默认子集：**en** 和 **zh**
- 被评估的 TTS 模型必须返回 `ContentAudio` 类型的生成音频，或在 completion 文本中返回音频路径、URL 或 data URI
- EvalScope 为 HTTP TTS 服务提供 `eval_type="text2speech"` 配置：
  - Volcengine 提供商：配置 `model="seed-tts-2.0"`、`api_url="https://openspeech.bytedance.com/api/v3/tts/unidirectional"` 和 `model_args={"speaker": "..."}`
  - OpenAI 提供商：配置 `model="tts-1"`（或 `tts-1-hd`）、`api_url="https://api.openai.com/v1"` 和 `model_args={"provider": "openai", "voice": "nova"}`
- 默认指标：**audio_wer**，该指标通过兼容 OpenAI 的 `/audio/transcriptions` 端点对生成的音频进行转录，并结合语言特定的归一化计算 WER/CER 风格的错误率
- 可通过 `metric_list` 配置 ASR 端点，或设置环境变量 `SEED_TTS_EVAL_ASR_API_BASE`、`SEED_TTS_EVAL_ASR_API_KEY`、`SEED_TTS_EVAL_ASR_MODEL` 和 `SEED_TTS_EVAL_ASR_API_PROTOCOL`
- 对于 Volcengine Ark 的 audio-understanding 模型，请设置 `api_protocol="responses"` 并使用支持音频输入的模型，例如 `doubao-seed-2-0-lite-260428`
- 说话人相似度是官方基准测试的一部分，但需要单独的说话人验证后端，默认未启用

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `seed_tts_eval` |
| **数据集ID** | [evalscope/Seed-TTS-Eval](https://modelscope.cn/datasets/evalscope/Seed-TTS-Eval/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2406.02430) |
| **标签** | `Audio`, `TextToSpeech` |
| **指标** | `audio_wer` |
| **默认样本数** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,108 |
| 提示词长度（平均） | 243.31 字符 |
| 提示词长度（最小/最大） | 193 / 376 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `en` | 1,088 | 304.3 | 224 | 376 |
| `zh` | 2,020 | 210.46 | 193 | 231 |

**音频统计数据：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 3,108 |
| 每样本音频数量 | 最小: 1, 最大: 1, 平均: 1 |
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

**提示模板：**
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
    datasets=['seed_tts_eval'],
    dataset_args={
        'seed_tts_eval': {
            # subset_list: ['en', 'zh']  # 可选，评估指定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```