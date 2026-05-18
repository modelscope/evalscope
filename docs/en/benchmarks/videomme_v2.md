# Video-MME-v2

## Overview

Video-MME-v2 is a public comprehensive video understanding benchmark for multimodal models. It contains 800 videos, 3,200 multiple-choice QA instances, and word-level subtitle files with timestamps.

This native adapter is intentionally built on the same reusable video benchmark path as MVBench: annotations are loaded through `DatasetHub`, optional media archives are resolved through the same hub abstraction, and samples use `ContentVideo` plus the OpenAI-compatible `video_url` path.

## Task Description

- **Task Type**: Video Multiple-Choice Question Answering
- **Input**: Video URL or official archived MP4 + question + answer choices
- **Output**: Single answer letter
- **Default Subset**: `all`
- **Full Benchmark**: 3,200 QA instances over 800 videos

## Key Features

- Public video benchmark with real video inputs
- Reuses `VisionLanguageAdapter`, `MultiChoiceAdapter`, `DatasetHub`, and `ContentVideo`
- Supports lightweight public-URL mode for small smoke tests
- Supports official MP4 archive mode through `extra_params.video_source = "archive"`
- Can include word-level subtitles in the prompt with `extra_params.use_subtitles = true`
- Supports subsets: `all`, `level_1`, `level_2`, `level_3`, `logic`, `relevance`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- The public ModelScope dataset stores official videos in 40 large zip archives; URL mode avoids downloading multi-GB archives for small tests
- Archive mode is more reproducible but can download several GB even for one sample

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `videomme_v2` |
| **Dataset ID** | [MME-Benchmarks/Video-MME-v2](https://modelscope.cn/datasets/MME-Benchmarks/Video-MME-v2) |
| **Paper** | [Video-MME-v2](https://arxiv.org/abs/2604.05015) |
| **Tags** | `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |

## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,200 |
| Videos | 800 |
| Questions per Video | 4 |

## Sample Example

```json
{
  "video_id": "001",
  "url": "https://www.youtube.com/watch?v=AYSYelOQtQI",
  "question_id": "001-1",
  "question": "What is the ethnicity of the protagonist's mother?",
  "options": "A. Malaysian.\nB. British.\nC. Singaporean.\nD. German.\nE. Canadian.\nF. Chinese.\nG. American.\nH. Cannot be determined.",
  "answer": "F"
}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets videomme_v2 \
    --dataset-args '{"videomme_v2": {"subset_list": ["all"], "extra_params": {"dataset_hub": "modelscope", "video_source": "url", "use_subtitles": true, "subtitle_word_limit": 512}}}' \
    --limit 2
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['videomme_v2'],
    dataset_args={
        'videomme_v2': {
            'subset_list': ['all'],
            'extra_params': {
                'dataset_hub': 'modelscope',
                'video_source': 'url',
                'use_subtitles': True,
                'subtitle_word_limit': 512,
            },
        }
    },
    limit=2,
)

run_task(task_cfg=task_cfg)
```
