# AIR-Bench-Chat


## Overview

AIR-Bench Chat is the generative half of [AIR-Bench](https://arxiv.org/abs/2402.07729) (Audio InstRuction Benchmark, ACL 2024 main conference) — the first instruction-following benchmark for large audio-language models (LALMs), covering **human speech, natural sounds and music**. It contains roughly 2k open-ended audio QA pairs covering speech, sound, music and mixed-audio scenes; responses are graded by a GPT-4 judge against a reference answer.

## Task Description

- **Task Type**: Open-ended audio question answering.
- **Input**: An audio clip plus a free-form question.
- **Output**: A textual answer evaluated against the reference response.

## Categories (8 tasks → 5 reported categories)

The 8 Chat tasks are aggregated by the official `cal_score.py` into five categories:

- `speech`: `speech_QA`, `speech_dialogue_QA`
- `sound`: `sound_QA`, `sound_generation_QA`
- `music`: `music_QA`, `music_generation_analysis_QA`
- `speech_and_sound`: `speech_and_sound_QA`
- `speech_and_music`: `speech_and_music_QA`

The paper's **Mixed-audio = mean(speech_and_sound, speech_and_music)**.

## Dataset Access

- The dataset is hosted on ModelScope: [`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset). It uses an *audiofolder + JSON metadata* layout. evalscope downloads it lazily via `modelscope.dataset_snapshot_download` on first run; the full release is ~49 GB, so it is recommended to limit which tasks are pulled via `extra_params`.
- If the dataset is already on disk, pass `dataset_args={'air_bench_chat': {'local_path': '/path/to/AIR-Bench-Dataset'}}`; the local root should contain `Chat/`.

## Evaluation Protocol

- The judge LLM (default: GPT-4) receives the question, the textual audio description (`meta_info` from the dataset), the reference answer (`answer_gt`), and the model's response. It outputs a single line with two integer scores in `[1, 10]`.
- To remove position bias, every sample is judged twice with the order of reference and prediction swapped, then averaged. This mirrors `cal_score.py` in the official repository — disable it via `extra_params={'do_swap': False}` to halve judge cost.
- Reported metric `gpt_score` is the model's mean judge score; `win_rate` records how often the model strictly beats the reference.

```{warning}
The official leaderboard uses `gpt-4-0125-preview` as the judge model. If that exact snapshot is unavailable, use an available GPT-4-class judge; absolute scores can drift versus the published numbers because the judge model changed.
```

## Implementation Notes

- The judge model is selected via `--judge-model-args`; ensure the model id supports long contexts (`meta_info` may exceed 4k tokens for dialogue tasks).
- Set `extra_params={'tasks': [...]}` to evaluate only specific Chat task names — useful for partial runs.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `air_bench_chat` |
| **Dataset ID** | [evalscope/AIR-Bench](https://modelscope.cn/datasets/evalscope/AIR-Bench/summary) |
| **Paper** | [Paper](https://aclanthology.org/2024.acl-long.109/) |
| **Tags** | `Audio`, `InstructionFollowing`, `QA` |
| **Metrics** | `gpt_score`, `win_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,200 |
| Prompt Length (Mean) | 83.89 chars |
| Prompt Length (Min/Max) | 17 / 423 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `speech_QA` | 400 | 64.33 | 23 | 148 |
| `speech_dialogue_QA` | 400 | 77.03 | 29 | 206 |
| `sound_QA` | 400 | 73.29 | 17 | 166 |
| `sound_generation_QA` | 100 | 222.52 | 130 | 423 |
| `music_QA` | 400 | 57.54 | 24 | 202 |
| `music_generation_analysis_QA` | 100 | 267.52 | 148 | 395 |
| `speech_and_sound_QA` | 200 | 63.98 | 25 | 127 |
| `speech_and_music_QA` | 200 | 69.37 | 32 | 127 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 2,200 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | mp3, wav |


## Sample Example

**Subset**: `speech_QA`

```json
{
  "input": [
    {
      "id": "5781ee73",
      "content": [
        {
          "audio": "/root/.cache/modelscope/hub/datasets/evalscope/AIR-Bench-Dataset/Chat/speech_QA_iemocap/Ses01F_script01_1_M025.wav",
          "format": "wav"
        },
        {
          "text": "Who is the speaker addressing at the end of the speech?"
        }
      ]
    }
  ],
  "target": "The speaker is addressing Mom at the end of the speech.",
  "id": 0,
  "group_id": 0,
  "subset_key": "speech_QA",
  "metadata": {
    "uniq_id": 400,
    "task_name": "speech_QA",
    "dataset_name": "iemocap",
    "category": "speech",
    "meta_info": "{'emotion': 'neutral', 'gender': 'male', 'transcription': \"And then we'll thrash it out with father. Okay Mom? Don't avoid me.\"}",
    "question": "Who is the speaker addressing at the end of the speech?"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `list` | `None` | Optional list of Chat task names to evaluate (subset of ['music_QA', 'music_generation_analysis_QA', 'sound_QA', 'sound_generation_QA', 'speech_QA', 'speech_and_music_QA', 'speech_and_sound_QA', 'speech_dialogue_QA']). Defaults to all tasks. |
| `do_swap` | `bool` | `True` | When True (default), each sample is judged twice with the order of reference vs. prediction swapped, then scores are averaged. Disable to halve judge cost at the price of position bias. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_chat \
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
    datasets=['air_bench_chat'],
    dataset_args={
        'air_bench_chat': {
            # subset_list: ['speech_QA', 'speech_dialogue_QA', 'sound_QA']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


