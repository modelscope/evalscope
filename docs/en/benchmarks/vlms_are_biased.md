# VLMs Are Biased


## Overview

VLMs Are Biased (VLMBias) tests whether vision-language models follow visual evidence when it conflicts with familiar concepts, such as a logo with an unusual number of stripes.

## Task Description

- **Task Type**: Visual question answering for counting and identification
- **Input**: A counterfactual or control image and a question
- **Output**: A number, `Yes`/`No`, or a short answer in curly brackets
- **Domain**: Animals, logos, flags, chess pieces, game boards, optical illusions, and patterned grids

## Key Features

- 11,594 image-question pairs across six splits; `main` contains 2,784 questions
- Seven topics and nineteen sub-topics, with identification, title-injection, original-image, and background-removal analyses
- Default data: **v2**, with revised optical-illusion images/questions, unique IDs, and a repaired identification image

## Evaluation Notes

- Metrics: **Accuracy** (`acc`, overall and by topic) and **Bias Ratio** (`bias_ratio`, lower is better), using official answer normalization and digit matching; `original` omits Bias Ratio
- All six splits run by default; set `subset_list=['main']` for the primary counting evaluation
- For the paper's v1 data, set `default_subset='v1-legacy'` in `dataset_args`. Use a new output directory when switching versions. The legacy `identification` split retains a truncated image
- Prompts are used verbatim. The official lmms-eval setup uses `temperature=0` and at most 32 new tokens
- [Paper](https://arxiv.org/abs/2505.23941) | [GitHub](https://github.com/anvo25/vlms-are-biased) | [Project page](https://vlmsarebiased.github.io/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `vlms_are_biased` |
| **Dataset ID** | [evalscope/vlms-are-biased](https://modelscope.cn/datasets/evalscope/vlms-are-biased/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2505.23941) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `accuracy`, `bias_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `main` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 11,594 |
| Prompt Length (Mean) | 88.3 chars |
| Prompt Length (Min/Max) | 60 / 130 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `main` | 2,784 | 89.97 | 78 | 129 |
| `identification` | 1,392 | 83.27 | 68 | 102 |
| `withtitle` | 2,784 | 89.97 | 78 | 129 |
| `original` | 458 | 85.03 | 60 | 130 |
| `remove_background_q1q2` | 2,784 | 90.19 | 78 | 125 |
| `remove_background_q3` | 1,392 | 83.91 | 70 | 102 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 11,594 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 384x183 - 1862x1430 |
| Formats | png |


## Sample Example

**Subset**: `main`

```json
{
  "input": [
    {
      "id": "a5b9d7f1",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~1.6KB]"
        },
        {
          "text": "Are the two red lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}."
        }
      ]
    }
  ],
  "target": "Yes",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "illusion-verticalhorizontal-001-diff0-notitle-q1-384",
    "legacy_id": "VerticalHorizontal_001_Q1_notitle_px384",
    "data_version": "v2",
    "topic": "Optical Illusion",
    "sub_topic": "Vertical-Horizontal illusion",
    "type_of_question": "Q1",
    "expected_bias": "No",
    "with_title": false,
    "pixel": 384
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vlms_are_biased \
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
    datasets=['vlms_are_biased'],
    dataset_args={
        'vlms_are_biased': {
            # subset_list: ['main', 'identification', 'withtitle']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
