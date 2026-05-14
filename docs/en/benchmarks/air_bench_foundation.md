# AIR-Bench-Foundation


## Overview

AIR-Bench Foundation is the discriminative half of [AIR-Bench](https://arxiv.org/abs/2402.07729) (Audio InstRuction Benchmark, ACL 2024 main conference) — the first instruction-following benchmark for large audio-language models (LALMs), covering **human speech, natural sounds and music**. The Foundation track contains roughly 25k single-choice questions spanning 19 logical tasks across three audio categories.

## Task Description

- **Task Type**: Single-choice question answering grounded on an audio clip.
- **Input**: One audio clip + a question with up to four candidate answers (A/B/C/D).
- **Output**: A single letter chosen from the provided options.

## Categories (19 tasks / 25 source-dataset subsets)

- **Speech** (11 dirs / 9 tasks): speech grounding, language ID, gender, emotion (IEMOCAP+MELD), age, speech entity recognition, intent classification, speaker counting, synthesized-voice detection.
- **Sound** (6 dirs / 4 tasks): audio grounding, vocal sound classification, acoustic scene classification (CochlScene+TUT2017), sound QA (avqa+clothoaqa).
- **Music** (8 dirs / 6 tasks): instruments, genre, MIDI pitch, MIDI velocity, music QA, music emotion.

## Prompt Template (matches official `Inference_Foundation.py`)

```text
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
```

## Dataset Access

- The dataset is hosted on ModelScope: [`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset). It uses an *audiofolder + JSON metadata* layout. evalscope downloads it lazily via `modelscope.dataset_snapshot_download` on first run; the full release is ~49 GB, so it is recommended to limit which subsets are pulled via `extra_params`.
- If the dataset is already on disk, pass `dataset_args={'air_bench_foundation': {'local_path': '/path/to/AIR-Bench-Dataset'}}`; the local root should contain `Foundation/`.
- Some Foundation samples are FLAC. For OpenAI-compatible audio input evalscope converts them to cached WAV files, which requires either `soundfile` (`pip install "evalscope[air_bench]"`) or a working `ffmpeg` binary.

## Evaluation Notes

- Metric: **accuracy** (per source-dataset subset, plus per-category aggregation).
- Default prompt follows the official `Inference_Foundation.py` formatting so existing AIR-Bench leaderboard numbers can be reproduced.
- Set `extra_params={'subsets': [...]}` to limit to a subset of the 25 source directories — useful for partial downloads.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `air_bench_foundation` |
| **Dataset ID** | [evalscope/AIR-Bench-Dataset](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset/summary) |
| **Paper** | [Paper](https://aclanthology.org/2024.acl-long.109/) |
| **Tags** | `Audio`, `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 7,189 |
| Prompt Length (Mean) | 234.43 chars |
| Prompt Length (Min/Max) | 185 / 321 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Acoustic_Scene_Classification_CochlScene` | 1,000 | 240.97 | 213 | 278 |
| `Acoustic_Scene_Classification_TUT2017` | 488 | 242.15 | 207 | 282 |
| `Audio_Grounding_AudioGrounding` | 896 | 273.74 | 249 | 321 |
| `Music_AQA_music_avqa` | 813 | 208.7 | 185 | 238 |
| `Music_Genre_Recognition_MTJ-Jamendo` | 1,000 | 223.84 | 200 | 248 |
| `Music_Genre_Recognition_fma` | 1,000 | 224.59 | 201 | 250 |
| `Music_Instruments_Classfication_MTJ-Jamendo` | 1,000 | 236.52 | 218 | 262 |
| `Music_Instruments_Classfication_nsynth` | 992 | 228.12 | 216 | 247 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 7,189 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | mp3, wav |


## Sample Example

**Subset**: `Speaker_Age_Prediction_common_voice_13.0_en`

```json
{
  "input": [
    {
      "id": "eb275a3a",
      "content": [
        {
          "audio": "/root/.cache/modelscope/hub/datasets/evalscope/AIR-Bench-Dataset/Foundation/Speaker_Age_Prediction_common_voice_13.0_en/common_voice_en_22159151.mp3",
          "format": "mp3"
        },
        {
          "text": "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.\nWhich age range do you believe best matches the speaker's voice?\nA. teens to twenties\nB. thirties to fourties\nC. fifties to sixties\nD. seventies to eighties"
        }
      ]
    }
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "Speaker_Age_Prediction_common_voice_13.0_en",
  "metadata": {
    "uniq_id": 5973,
    "task_name": "Speaker_Age_Prediction",
    "dataset_name": "common_voice_13.0_en",
    "category": "speech",
    "answer_gt_text": "thirties to fourties",
    "choices": {
      "A": "teens to twenties",
      "B": "thirties to fourties",
      "C": "fifties to sixties",
      "D": "seventies to eighties"
    }
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subsets` | `list` | `None` | Optional list of Foundation source-dataset directories to evaluate. Defaults to all 25 directories. Useful when only a subset has been downloaded locally. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_foundation \
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
    datasets=['air_bench_foundation'],
    dataset_args={
        'air_bench_foundation': {
            # subset_list: ['Acoustic_Scene_Classification_CochlScene', 'Acoustic_Scene_Classification_TUT2017', 'Audio_Grounding_AudioGrounding']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


