# Seed-TTS-Eval


## Overview

Seed-TTS-Eval is an objective benchmark for zero-shot text-to-speech and voice conversion evaluation. It uses out-of-domain English and Mandarin samples from Common Voice and DiDiSpeech-2, and the official evaluation focuses on intelligibility and speaker consistency.

## Task Description

- **Task Type**: Zero-shot text-to-speech generation
- **Input**: Reference speaker audio, prompt transcript, and target text
- **Output**: Synthesized speech audio for the target text using the reference speaker
- **Languages**: English and Mandarin

## Evaluation Notes

- Default subsets: **en** and **zh**
- The evaluated TTS model must return generated audio as `ContentAudio`, or return an audio path, URL, or data URI as the completion text
- EvalScope provides `eval_type="text2speech"` for HTTP TTS services.
  - Volcengine provider: configure `model="seed-tts-2.0"`, `api_url="https://openspeech.bytedance.com/api/v3/tts/unidirectional"`, and `model_args={"speaker": "..."}`
  - OpenAI provider: configure `model="tts-1"` (or `tts-1-hd`), `api_url="https://api.openai.com/v1"`, and `model_args={"provider": "openai", "voice": "nova"}`
- Default metric: **audio_wer**, which transcribes generated audio through an OpenAI-compatible `/audio/transcriptions` endpoint and computes WER/CER-style error rate with language-specific normalization
- Configure the ASR endpoint via `metric_list`, or set `SEED_TTS_EVAL_ASR_API_BASE`, `SEED_TTS_EVAL_ASR_API_KEY`, `SEED_TTS_EVAL_ASR_MODEL`, and `SEED_TTS_EVAL_ASR_API_PROTOCOL`
- For Volcengine Ark audio-understanding models, set `api_protocol="responses"` and use a model that supports audio input, such as `doubao-seed-2-0-lite-260428`
- Speaker similarity is part of the official benchmark, but it requires a separate speaker verification backend and is not enabled by default


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `seed_tts_eval` |
| **Dataset ID** | [evalscope/Seed-TTS-Eval](https://modelscope.cn/datasets/evalscope/Seed-TTS-Eval/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2406.02430) |
| **Tags** | `Audio`, `TextToSpeech` |
| **Metrics** | `audio_wer` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,108 |
| Prompt Length (Mean) | 243.31 chars |
| Prompt Length (Min/Max) | 193 / 376 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en` | 1,088 | 304.3 | 224 | 376 |
| `zh` | 2,020 | 210.46 | 193 | 231 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 3,108 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | wav |


## Sample Example

**Subset**: `en`

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

## Prompt Template

**Prompt Template:**
```text
Use the reference audio and prompt transcript to synthesize the target text in the same speaker voice.
Prompt transcript: {prompt_text}
Target text: {text}
Return only the generated audio.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets seed_tts_eval \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

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


