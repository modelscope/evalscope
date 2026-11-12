# VLMs Are Biased


## Overview

VLMs Are Biased (VLMBias) evaluates whether vision-language models answer objective visual questions from the image or fall back to memorized prior knowledge. It uses counterfactual images whose visible properties conflict with familiar concepts, such as an Adidas-style logo with four stripes or an animal with an unusual number of legs.

## Task Description

- **Task Type**: Free-form visual question answering for counting and identification
- **Input**: A counterfactual or control image paired with a counting, binary identification, or short-answer question
- **Output**: A number, `Yes`/`No`, or a short identity enclosed in curly brackets
- **Domain**: Animals, logos, flags, chess pieces, game boards, optical illusions, and patterned grids

## Key Features

- The primary `main` split contains 2,784 objective visual questions over 1,392 counterfactual images at 384, 768, and 1152 pixel resolutions
- Five official analysis splits cover binary identification, in-image title injection, original unmodified controls, and background-removed variants
- Each counterfactual record provides both the visually correct `ground_truth` and the prior-knowledge `expected_bias`
- The benchmark exposes seven topics and nineteen sub-topics for detailed analysis without creating synthetic EvalScope subsets
- The default data is **v2**: optical-illusion target features are red and named in the questions, IDs are globally unique, and the truncated identification image is repaired
- Both hubs are pinned to matching v2 snapshots; sample metadata preserves the v2 `ID`, the original `legacy_ID`, and the selected data version

## Evaluation Notes

- The dataset prompt is used verbatim, including its required curly-bracket answer format
- Primary metric: **Accuracy** (`acc`), using the official case-insensitive comparison after stripping outer braces; if exact text matching fails, digit sequences are compared
- Secondary metric: **Bias Ratio** (`bias_ratio`, lower is better), the fraction of predictions matching `expected_bias` under the same normalization
- Accuracy is also reported by topic, matching the official lmms-eval integration
- `bias_ratio` is omitted for the `original` split because those control records do not define `expected_bias`
- The six official dataset splits are evaluated as separate EvalScope subsets by default; select only `main` for the primary counting evaluation
- Default v2 runs use evaluation version **v1.1**. For the paper's v1 data, explicitly set `dataset_args={'vlms_are_biased': {'default_subset': 'v1-legacy', 'subset_list': ['main']}}`; these runs are labeled **v1 legacy** and use evaluation version **v1.0**. The legacy `identification` split retains the upstream truncated-image defect
- v1 and v2 results are not directly interchangeable. Start a new output directory after changing data versions; do not use `rerun_review` to reuse predictions for changed images or prompts. Historical unpinned runs require inspection of their saved inputs/IDs to establish the data version
- Generation should be deterministic and concise; the official lmms-eval setup uses `temperature=0` and at most 32 new tokens
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
      "id": "0d71a23a",
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
