# MVBench

## Overview

MVBench is a public video understanding benchmark for multimodal models. It evaluates temporal perception, attribute/state reasoning, symbolic ordering, and high-level cognition through video multiple-choice questions. This native adapter uses the `PKU-Alignment/MVBench` ModelScope mirror by default, which provides the original annotations plus optimized video archives.

## Task Description

- **Task Type**: Video Multiple-Choice Question Answering
- **Input**: Video + question + answer choices
- **Output**: Single answer letter
- **Default Subset**: `action_antonym` for lightweight smoke testing
- **Full Benchmark**: 20 subsets, 200 samples per subset

## Key Features

- Public video benchmark with real MP4 inputs
- Uses existing `VisionLanguageAdapter` and `MultiChoiceAdapter` flow
- Loads annotations and video archives through the shared `dataset_hub` abstraction
- Lazily extracts only the videos referenced by the selected samples
- Sends local videos through the OpenAI-compatible `video_url` content path
- Preserves optional `start`/`end`/`fps` video metadata and adds a segment instruction when a record is time-bounded
- Supports selecting MVBench subsets through `dataset_args`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- The default `action_antonym` subset downloads the small `ssv2_video.zip` archive and is suitable for CI-style validation
- Full MVBench evaluation requires downloading the corresponding public video archives for selected subsets
- The adapter defaults to ModelScope for better availability in China; Hugging Face or local mirrors can be selected with `extra_params.dataset_hub` and `extra_params.dataset_id`

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mvbench` |
| **Dataset ID** | [PKU-Alignment/MVBench](https://modelscope.cn/datasets/PKU-Alignment/MVBench) |
| **Paper** | [MVBench](https://arxiv.org/abs/2311.17005) |
| **Tags** | `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |

## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,000 |
| Subsets | 20 |
| Samples per Subset | 200 |

## Supported Subsets

`action_antonym`, `action_count`, `action_localization`, `action_prediction`, `action_sequence`, `character_order`, `counterfactual_inference`, `egocentric_navigation`, `episodic_reasoning`, `fine_grained_action`, `fine_grained_pose`, `moving_attribute`, `moving_count`, `moving_direction`, `object_existence`, `object_interaction`, `object_shuffle`, `scene_transition`, `state_change`, `unexpected_action`.

## Sample Example

```json
{
  "video": "166583.mp4",
  "question": "What is the action performed by the person in the video?",
  "candidates": [
    "Not sure",
    "Scattering something down",
    "Piling something up"
  ],
  "answer": "Piling something up"
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mvbench \
    --dataset-args '{"mvbench": {"subset_list": ["action_antonym"], "extra_params": {"dataset_hub": "modelscope"}}}' \
    --limit 10
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['mvbench'],
    dataset_args={
        'mvbench': {
            'subset_list': ['action_antonym'],
            'extra_params': {
                'dataset_hub': 'modelscope',
            },
        }
    },
    limit=10,
)

run_task(task_cfg=task_cfg)
```
