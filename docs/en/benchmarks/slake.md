# SLAKE


## Overview

SLAKE is a bilingual (English / Chinese) radiology visual question answering benchmark built by
physicians on CT, MRI and X-Ray images. Questions cover both purely visual properties of the scan
and medical knowledge that has to be recalled on top of what the image shows.

## Task Description

- **Task Type**: Medical visual question answering (free-form short answer)
- **Input**: A radiology image plus a question in English or Chinese
- **Output**: A single word or short phrase, in the language of the question
- **Domain**: Radiology (chest, abdomen, brain, pelvis, neck)

## Key Features

- 2,094 test questions over 180 images, roughly balanced between English (1,061) and Chinese (1,033)
- Every question is labelled `OPEN` (free answer) or `CLOSED` (answer drawn from a small closed set,
  mostly yes/no), which is the breakdown the original paper reports
- Questions span ten semantic types: organ, position, abnormality, knowledge-graph, modality, size,
  plane, quantity, color and shape
- Knowledge-graph questions (`base_type=kvqa`) ask about causes, symptoms, treatments and functions
  that cannot be read off the image

## Evaluation Notes

- Primary metric: **Accuracy** by normalized exact match against the single reference answer
- Reported as four subsets, `<language>_<open|closed>`, grouped into an English and a Chinese
  category; the overall score is the sample-weighted mean
- Normalization lower-cases, removes punctuation and parenthesised asides, maps yes/no synonyms
  onto one label — required because Chinese references express the same polarity as
  是的 / 有 / 包含 / 可以 or 不是 / 没有 / 不包含 / 不可以 — and unifies the X-Ray spellings, including
  the Chinese X光 / X射线, because modality references stay in English in the Chinese half
- Answers are read from the `ANSWER:` line requested by the prompt; when the model does not emit
  one, the whole reply is normalized instead, so a reply that only restates the question scores 0
- Exact match is strict by design, matching the original classification-style evaluation: a
  reference such as `Lung, Spinal Cord`, a knowledge-graph list of treatments, or `T2` answered as
  `T2-weighted` only counts when the model reproduces the reference wording, so open-ended
  accuracy on the knowledge-graph questions is expected to be low
- Images ship as a single `imgs.zip` (about 200 MB) and are read directly from the archive
- [Paper](https://arxiv.org/abs/2102.09542) | [Project page](https://www.med-vqa.com/slake/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `slake` |
| **Dataset ID** | [evalscope/SLAKE](https://modelscope.cn/datasets/evalscope/SLAKE/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2102.09542) |
| **Tags** | `Medical`, `MultiModal`, `QA` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,094 |
| Prompt Length (Mean) | 130.2 chars |
| Prompt Length (Min/Max) | 60 / 257 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en_open` | 645 | 195.09 | 168 | 257 |
| `en_closed` | 416 | 187.99 | 162 | 253 |
| `zh_open` | 613 | 67.07 | 61 | 79 |
| `zh_closed` | 420 | 65.44 | 60 | 82 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,094 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 240x240 - 1024x1024 |
| Formats | jpeg |


## Sample Example

**Subset**: `en_open`

```json
{
  "input": [
    {
      "id": "411b63eb",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~63.2KB]"
        },
        {
          "text": "What modality is used to take this image?\nAnswer the question with a single word or phrase in English.\nThe last line of your response must be of the form \"ANSWER: <answer>\" (without quotes)."
        }
      ]
    }
  ],
  "target": "CT",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "qid": 11934,
    "img_name": "xmlab102/source.jpg",
    "answer_type": "OPEN",
    "content_type": "Modality",
    "modality": "CT"
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
    --datasets slake \
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
    datasets=['slake'],
    dataset_args={
        'slake': {
            # subset_list: ['en_open', 'en_closed', 'zh_open']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
