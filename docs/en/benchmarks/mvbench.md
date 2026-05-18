# MVBench


## Overview

MVBench is a public multimodal video understanding benchmark covering temporal perception,
attribute/state reasoning, symbolic ordering, and high-level cognition. This native adapter uses
the ModelScope `PKU-Alignment/MVBench` mirror by default, which provides JSON annotations plus
optimized video archives.

## Task Description

- **Task Type**: Video multiple-choice question answering
- **Input**: Video + question + answer choices
- **Output**: Single correct answer letter
- **Subsets**: 20 MVBench tasks; the default smoke-test subset is `action_antonym`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- The default `action_antonym` subset downloads a small public MP4 archive for quick validation
- Full benchmark evaluation can be requested by setting `subset_list` to additional MVBench subsets
- Time-bounded records keep start/end metadata and add a short segment instruction to the prompt


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mvbench` |
| **Dataset ID** | [PKU-Alignment/MVBench](https://modelscope.cn/datasets/PKU-Alignment/MVBench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2311.17005) |
| **Tags** | `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,000 |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `action_antonym` | 200 | N/A | N/A | N/A |
| `action_count` | 200 | N/A | N/A | N/A |
| `action_localization` | 200 | N/A | N/A | N/A |
| `action_prediction` | 200 | N/A | N/A | N/A |
| `action_sequence` | 200 | N/A | N/A | N/A |
| `character_order` | 200 | N/A | N/A | N/A |
| `counterfactual_inference` | 200 | N/A | N/A | N/A |
| `egocentric_navigation` | 200 | N/A | N/A | N/A |
| `episodic_reasoning` | 200 | N/A | N/A | N/A |
| `fine_grained_action` | 200 | N/A | N/A | N/A |
| `fine_grained_pose` | 200 | N/A | N/A | N/A |
| `moving_attribute` | 200 | N/A | N/A | N/A |
| `moving_count` | 200 | N/A | N/A | N/A |
| `moving_direction` | 200 | N/A | N/A | N/A |
| `object_existence` | 200 | N/A | N/A | N/A |
| `object_interaction` | 200 | N/A | N/A | N/A |
| `object_shuffle` | 200 | N/A | N/A | N/A |
| `scene_transition` | 200 | N/A | N/A | N/A |
| `state_change` | 200 | N/A | N/A | N/A |
| `unexpected_action` | 200 | N/A | N/A | N/A |

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
| `dataset_id` | `str` | `PKU-Alignment/MVBench` | Dataset repository ID or local dataset root for MVBench annotations and videos. |
| `dataset_hub` | `str` | `modelscope` | Dataset hub used to load annotations and video archives. Choices: ['huggingface', 'modelscope', 'local'] |
| `dataset_revision` | `str` | `` | Optional dataset revision; leave empty to use the hub default. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mvbench \
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
    datasets=['mvbench'],
    dataset_args={
        'mvbench': {
            # subset_list: ['action_antonym', 'action_count', 'action_localization']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


