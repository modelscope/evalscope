# AIR-Bench

## Overview

[AIR-Bench](https://arxiv.org/abs/2402.07729) (Audio InstRuction Benchmark, ACL 2024 main conference) is the first instruction-following benchmark for large audio-language models (LALMs). It covers **human speech, natural sounds and music**, and supports direct grading of free-form responses.

AIR-Bench is split into two tracks, registered separately in evalscope:

| Benchmark name | Task type | Size | Scoring |
|---|---|---|---|
| `air_bench_foundation` | Single-choice MCQ (A/B/C/D) | ~25k | Rule-based accuracy |
| `air_bench_chat` | Open-ended audio QA | ~2k | GPT-4 judge (1–10) with position-swap averaging |

## Dataset Access

The dataset is hosted on Hugging Face: [`qyang1021/AIR-Bench-Dataset`](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset). It uses an *audiofolder + JSON metadata* layout. evalscope downloads it lazily via `huggingface_hub.snapshot_download` on first run; the full release is ~49 GB, so it is recommended to limit which subsets are pulled via `extra_params`.

If the dataset is already on disk, pass `dataset_args={'<benchmark>': {'local_path': '/path/to/AIR-Bench-Dataset'}}`; the local root should contain `Foundation/` and/or `Chat/`.

Some Foundation samples are FLAC. For OpenAI-compatible audio input evalscope converts them to cached WAV files, which requires either `soundfile` (`pip install "evalscope[air_bench]"`) or a working `ffmpeg` binary.

## Foundation track (`air_bench_foundation`)

19 logical tasks distributed over 25 source-dataset directories, aggregated into three categories per the paper:

- **Speech** (11 dirs / 9 tasks): speech grounding, language ID, gender, emotion (IEMOCAP+MELD), age, speech entity, intent, speaker counting, synthesized voice.
- **Sound** (6 dirs / 4 tasks): audio grounding, vocal sound, acoustic scene (CochlScene+TUT2017), sound QA (avqa+clothoaqa).
- **Music** (8 dirs / 6 tasks): instruments, genre, MIDI pitch, MIDI velocity, music QA, music emotion.

### Properties

| Property | Value |
|---|---|
| **Benchmark name** | `air_bench_foundation` |
| **Dataset ID** | [qyang1021/AIR-Bench-Dataset](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset) |
| **Paper** | [AIR-Bench (Yang et al., ACL 2024)](https://aclanthology.org/2024.acl-long.109.pdf) |
| **Tags** | `Audio`, `MCQ`, `Knowledge` |
| **Metrics** | `acc` |
| **Evaluation split** | `test` |
| **Extra params** | `subsets`: optional list of source directories to evaluate |

### Prompt template (matches official `Inference_Foundation.py`)

```text
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
```

### CLI usage

```bash
evalscope eval \
    --model YOUR_AUDIO_LLM \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_foundation \
    --limit 10  # remove for a full run
```

### Python usage (music subsets only)

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_AUDIO_LLM',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['air_bench_foundation'],
    dataset_args={
        'air_bench_foundation': {
            # 'local_path': '/path/to/AIR-Bench-Dataset',  # optional
            'extra_params': {
                'subsets': [
                    'Music_AQA_music_avqa',
                    'Music_Genre_Recognition_MTJ-Jamendo',
                    'Music_Genre_Recognition_fma',
                    'Music_Instruments_Classfication_MTJ-Jamendo',
                    'Music_Instruments_Classfication_nsynth',
                    'Music_Midi_Pitch_Analysis_nsynth',
                    'Music_Midi_Velocity_Analysis_nsynth',
                    'Music_Mood_Recognition_MTJ-Jamendo',
                ]
            }
        }
    },
)

run_task(task_cfg=task_cfg)
```

## Chat track (`air_bench_chat`)

8 tasks aggregated by `cal_score.py` into five categories: `speech`, `sound`, `music`, `speech_and_sound`, `speech_and_music`. The paper's **Mixed-audio = mean(speech_and_sound, speech_and_music)**.

### Properties

| Property | Value |
|---|---|
| **Benchmark name** | `air_bench_chat` |
| **Dataset ID** | [qyang1021/AIR-Bench-Dataset](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset) |
| **Paper** | [AIR-Bench (Yang et al., ACL 2024)](https://aclanthology.org/2024.acl-long.109.pdf) |
| **Tags** | `Audio`, `QA`, `InstructionFollowing` |
| **Metrics** | `gpt_score` (mean 1–10 judge score), `win_rate` (fraction of strict wins over the reference) |
| **Evaluation split** | `test` |
| **Extra params** | `tasks`: optional list of task names to evaluate; `do_swap`: enable position-swap averaging (default `True`) |

### Scoring protocol

The judge prompt mirrors the official template: it asks the LLM to score Assistant 1 (reference) and Assistant 2 (model output) on a 1–10 scale. To remove position bias, every sample is judged twice, with reference and prediction swapped between passes; the two passes are averaged.

```{warning}
The official leaderboard uses `gpt-4-0125-preview` as the judge model. If that exact snapshot is unavailable, use an available GPT-4-class judge; absolute scores can drift versus the published numbers because the judge model changed.
```

### CLI usage (with judge-model config)

```bash
evalscope eval \
    --model YOUR_AUDIO_LLM \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_chat \
    --judge-strategy llm \
    --judge-model-args '{"model_id": "YOUR_GPT4_JUDGE", "api_key": "OPENAI_KEY"}' \
    --limit 10
```

### Python usage (single-pass judging to halve cost)

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_AUDIO_LLM',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['air_bench_chat'],
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_GPT4_JUDGE',
        'api_key': 'OPENAI_KEY',
    },
    dataset_args={
        'air_bench_chat': {
            # 'local_path': '/path/to/AIR-Bench-Dataset',  # optional
            'extra_params': {
                'tasks': ['speech_QA', 'speech_dialogue_QA'],
                'do_swap': False,
            }
        }
    },
)

run_task(task_cfg=task_cfg)
```
