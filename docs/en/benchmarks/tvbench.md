# TVBench


## Overview

TVBench is a temporal video understanding benchmark for evaluating whether multimodal models can reason over dynamic visual events rather than isolated frames. It covers a broad set of video reasoning skills, including action recognition, action counting, temporal localization, action order understanding, egocentric action sequencing, object counting, object shuffling, moving direction recognition, scene transition reasoning, and unexpected action detection.

The EvalScope native adapter reads the official per-task JSON annotations from the dataset repository and resolves the corresponding video archives on demand. This keeps the default smoke-test path lightweight while still supporting the full benchmark through `subset_list`.

## Task Description

- **Task Type**: Video multiple-choice question answering (MCQ)
- **Input**: A video clip or a time-bounded video segment, a natural-language question, and 2-4 answer candidates
- **Output**: A single answer option letter selected from the provided candidates
- **Default Subset**: `action_count`, selected because it is available as a standard MP4 archive in the public dataset repository
- **Supported Subsets**: `action_antonym`, `action_count`, `action_localization`, `action_sequence`, `egocentric_sequence`, `moving_direction`, `object_count`, `object_shuffle`, `scene_transition`, and `unexpected_action`

## Evaluation Notes

- Default evaluation uses 0-shot Chain-of-Thought multiple-choice prompting via `MultipleChoiceTemplate.SINGLE_ANSWER_COT`.
- Primary metric: Accuracy (`acc`). The dataset answer is stored as candidate text and is converted to the corresponding option letter before scoring.
- Some subsets provide `start`/`end` fields. The adapter passes these values to `ContentVideo` and also adds a concise segment instruction to the prompt.
- Video files are downloaded lazily from subset-specific archives. `egocentric_sequence` uses segmented archives under `video/egocentric_sequence/<prefix>.zip`.
- The `action_antonym` annotations reference AVI files. If the repository media archive is unavailable, configure `extra_params.video_dir` to a local directory containing the AVI files.
- The adapter supports local video layouts that are either flat (`video_dir/<video>`) or grouped by subset (`video_dir/<subset>/<video>`).


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tvbench` |
| **Dataset ID** | [evalscope/TVBench](https://modelscope.cn/datasets/evalscope/TVBench/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 536 |
| Prompt Length (Mean) | 326.44 chars |
| Prompt Length (Min/Max) | 290 / 358 chars |

**Video Statistics:**

| Metric | Value |
|--------|-------|
| Total Videos | 536 |
| Videos per Sample | min: 1, max: 1, mean: 1 |
| Formats | mp4 |


## Sample Example

**Subset**: `action_count`

```json
{
  "input": [
    {
      "id": "4b521c81",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nThe person makes sets of repeated actions. How many times did the person repeat the action in the last set?\n\nA) 2\nB) 3\nC) 5\nD) 4"
        },
        {
          "video": "~/.cache/modelscope/hub/datasets/tvbench/videos/action_count/video_8686.mp4",
          "format": "mp4"
        }
      ]
    }
  ],
  "choices": [
    "2",
    "3",
    "5",
    "4"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "video": "video_8686.mp4",
    "answer": "4",
    "subset": "action_count",
    "start": null,
    "end": null,
    "fps": null,
    "accurate_start": null,
    "accurate_end": null,
    "video_length": null,
    "question_id": null,
    "dataset_id": "evalscope/TVBench",
    "dataset_hub": "modelscope"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_dir` | `str` | `` | Optional local directory containing TVBench video files. It may be organized flat or by subset. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tvbench \
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
    datasets=['tvbench'],
    dataset_args={
        'tvbench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
