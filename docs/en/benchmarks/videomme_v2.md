# Video-MME-v2


## Overview

Video-MME-v2 is a public comprehensive video understanding benchmark. It contains 800 videos,
3,200 multiple-choice QA instances, and word-level subtitles with timestamps. The native adapter
uses the shared `DatasetHub` abstraction for both annotation loading and optional media archive
downloads, so it exercises the same reusable video benchmark path as MVBench.

## Task Description

- **Task Type**: Video multiple-choice question answering
- **Input**: Video URL or archived MP4 + question + answer choices
- **Output**: Single correct answer letter
- **Subsets**: `all`, `level_1`, `level_2`, `level_3`, `logic`, `relevance`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- The default video source is the public `url` field for lightweight smoke tests
- Set `extra_params.video_source` to `archive` to download and use the official MP4 archives
- Set `extra_params.use_subtitles` to `true` to include word-level subtitles in the prompt


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `videomme_v2` |
| **Dataset ID** | [MME-Benchmarks/Video-MME-v2](https://modelscope.cn/datasets/MME-Benchmarks/Video-MME-v2/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2604.05015) |
| **Tags** | `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,200 |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `all` | 3,200 | N/A | N/A | N/A |
| `level_1` | 686 | N/A | N/A | N/A |
| `level_2` | 834 | N/A | N/A | N/A |
| `level_3` | 837 | N/A | N/A | N/A |
| `logic` | 1,124 | N/A | N/A | N/A |
| `relevance` | 2,076 | N/A | N/A | N/A |

## Sample Example

*Sample example not available.*

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
| `dataset_id` | `str` | `MME-Benchmarks/Video-MME-v2` | Dataset repository ID or local dataset root for Video-MME-v2. |
| `dataset_hub` | `str` | `modelscope` | Dataset hub used to load annotations, subtitles, and optional video archives. Choices: ['huggingface', 'modelscope', 'local'] |
| `dataset_revision` | `str` | `` | Optional dataset revision; leave empty to use the hub default. |
| `video_source` | `str` | `url` | Use public URL fields for lightweight tests or official archived MP4 files. Choices: ['url', 'archive'] |
| `use_subtitles` | `bool` | `False` | Include Video-MME-v2 subtitle text in the prompt. |
| `subtitle_word_limit` | `int` | `512` | Maximum number of subtitle words included per sample when subtitles are enabled. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets videomme_v2 \
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
    datasets=['videomme_v2'],
    dataset_args={
        'videomme_v2': {
            # subset_list: ['all', 'level_1', 'level_2']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


