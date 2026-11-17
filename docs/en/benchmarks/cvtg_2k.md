# CVTG-2K


## Overview

CVTG-2K is a benchmark of 2,000 prompts for evaluating complex visual text generation
(TextCrafter, arXiv:2503.23461). Prompts contain 2-5 quoted text regions that the generated
image must render; prompts are organized by region count under the CVTG and CVTG-Style splits
(the latter annotates style attributes such as size, color and font).

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation (text rendering)
- **Input**: Text prompt with quoted text regions
- **Output**: Generated image scored with Word Accuracy and word-level NED against the quoted text
- **Domain**: Complex visual text generation for text-to-image models

## Key Features

- 2,000 prompts (CVTG and CVTG-Style splits, 2-5 text regions each) from the official
  TextCrafter release
- Scores Word Accuracy and word-level NED against the single-quoted fragments of each prompt,
  mirroring the official TextCrafter evaluator including its corpus-level aggregation
- Shares one PaddleOCR engine across both metrics per language

## Evaluation Notes

- Ground-truth words are the single-quoted fragments of each prompt, matching the official
  TextCrafter evaluator; repeated ground-truth words are not consumed
- Metrics: cvtg_word_accuracy (membership word match) and cvtg_ned (word-level best-match
  Levenshtein similarity, higher is better), both aggregated corpus-level (total numerator /
  total denominator) as in the official evaluator
- Loads the complete 2,000-prompt ModelScope mirror by default. The CVTG and CVTG-Style
  configurations contain 1,002 and 998 test examples, respectively; the original source is
  [dnkdnk/CVTG-2K](https://huggingface.co/datasets/dnkdnk/CVTG-2K).
- Install PaddleOCR 2.x and PaddlePaddle for OCR scoring (for example, paddleocr==2.10.0
  and paddlepaddle==2.6.2). Missing dependencies or OCR failures exclude the affected
  metric instead of assigning a zero. A valid image with no recognized text counts as missing text.
- Images can be generated with a local diffusion model or an OpenAI-compatible image
  generation service; set model_task='image_generation' and eval_type='text2image'.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cvtg_2k` |
| **Dataset ID** | [evalscope/CVTG-2K](https://modelscope.cn/datasets/evalscope/CVTG-2K/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2503.23461) |
| **Tags** | `TextToImage` |
| **Metrics** | `CVTGWordAccuracy`, `CVTGNED` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,000 |
| Prompt Length (Mean) | 195.92 chars |
| Prompt Length (Min/Max) | 67 / 462 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `CVTG` | 1,002 | 161.05 | 67 | 343 |
| `CVTG-Style` | 998 | 230.93 | 106 | 462 |

## Sample Example

**Subset**: `CVTG`

```json
{
  "input": [
    {
      "id": "dd076101",
      "content": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "prompt": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'.",
    "carrier_list": [
      "sign",
      "cup"
    ],
    "sentence_list": [
      "a sign that says 'READ'.",
      "a coffee cup on the table with the word 'MORNING'."
    ],
    "benchmark_type": "CVTG",
    "region_count": 2,
    "category": "CVTG"
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
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets cvtg_2k \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['cvtg_2k'],
    dataset_args={
        'cvtg_2k': {
            # subset_list: ['CVTG', 'CVTG-Style']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
