# Seed-TTS-Eval

Seed-TTS-Eval 用于零样本文本转语音（TTS）和语音转换任务的客观评测基准。
评测数据来自 Common Voice 与 DiDiSpeech-2 的越域英文和中文样本，
官方评测重点关注可懂度（intelligibility）和说话人一致性（speaker consistency）。

## 任务说明

- **任务类型**：零样本文本转语音生成
- **输入**：参考说话人音频、参考转录文本、目标文本
- **输出**：使用参考说话人音色合成目标文本对应音频
- **语言**：英语、中文

## 评测说明

- 默认子集：**en** 和 **zh**
- 评测中的 TTS 模型需要返回 `ContentAudio`，或返回音频路径、URL、或 data URI 作为 completion 文本
- EvalScope 为 HTTP TTS 服务提供 `eval_type="text2speech"`：
  - Volcengine：
    - `model="seed-tts-2.0"`
    - `api_url="https://openspeech.bytedance.com/api/v3/tts/unidirectional"`
    - `model_args={"speaker": "..."}`
  - OpenAI：
    - `model="tts-1"`（或 `tts-1-hd`）
    - `api_url="https://api.openai.com/v1"`
    - `model_args={"provider": "openai", "voice": "nova"}`
- 默认指标为 **audio_wer**，通过 OpenAI 兼容的 `/audio/transcriptions` 端点转写音频，并在语言归一化后计算 WER/CER 风格误差
- 也可通过 `metric_list` 配置 ASR 端点，或设置以下环境变量：
  - `SEED_TTS_EVAL_ASR_API_BASE`
  - `SEED_TTS_EVAL_ASR_API_KEY`
  - `SEED_TTS_EVAL_ASR_MODEL`
  - `SEED_TTS_EVAL_ASR_API_PROTOCOL`
- 对于 Volcengine Ark 音频理解模型，可设置 `api_protocol="responses"` 并使用支持音频输入的模型（如 `doubao-seed-2-0-lite-260428`）
- 说话人相似度是官方评测的一部分，但当前未默认开启；该能力需要独立的说话人验证后端

## 属性

| 字段 | 说明 |
|----------|-------|
| **任务名称** | `seed_tts_eval` |
| **数据集 ID** | [TwinkStart/Seed-TTS-Eval](https://modelscope.cn/datasets/TwinkStart/Seed-TTS-Eval/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2406.02430) |
| **标签** | `Audio`, `TextToSpeech` |
| **指标** | `audio_wer` |
| **默认 shots** | `0-shot` |
| **评测 Split** | `train` |

## 数据统计

| 指标 | 数值 |
|--------|-------|
| 样本总量 | 3,108 |
| Prompt 平均长度 | 243.31 chars |
| Prompt 长度（最小/最大） | 193 / 376 chars |

**各子集统计：**

| 子集 | 样本数 | Prompt 平均长度 | Prompt 最小值 | Prompt 最大值 |
|--------|---------|-------------|------------|------------|
| `en` | 1,088 | 304.3 | 224 | 376 |
| `zh` | 2,020 | 210.46 | 193 | 231 |

**音频统计：**

| 指标 | 数值 |
|--------|-------|
| 音频文件总量 | 3,108 |
| 每条样本音频数 | min: 1, max: 1, mean: 1 |
| 格式 | wav |

## 示例

**子集**：`en`

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
          "text": "Use reference audio to synthesize target text."
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

## Prompt 模板

**Prompt Template：**
```text
Use the reference audio and prompt transcript to synthesize the target text in the same speaker voice.
Prompt transcript: {prompt_text}
Target text: {text}
Return only the generated audio.
```

## 额外参数

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `dataset_hub` | `str` | `huggingface` | 指定加载 Seed-TTS-Eval 的数据源，可选：`huggingface`、`modelscope`、`local` |

## 使用方式

### 命令行

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets seed_tts_eval \
    --limit 10  # 正式评测请移除该行
```

### Python

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
            # subset_list: ['en', 'zh']  # 可选：评测指定子集
            # extra_params: {}  # 使用默认 extra_params 时可省略
        }
    },
    limit=10,  # 正式评测请移除该行
)

run_task(task_cfg=task_cfg)
```
