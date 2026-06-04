# Maritime-OCR-Bench


## Overview

Maritime-OCR-Bench is a comprehensive evaluation benchmark for assessing multimodal large model capabilities
on OCR-related tasks. The current released set contains 1,888 manually curated samples across five task types.

## Task Types

- **VQA**: Visual question answering on document/scene images
- **IE**: Information extraction requiring strict JSON output
- **parsing**: Text recognition and parsing from images
- **json1**: Text spotting with JSON v1 structured output
- **json2**: Text spotting with JSON v2 structured output

## Evaluation Metrics

Each task type uses a specialized scoring method:
- VQA/parsing: Multi-dimensional text similarity (edit distance, char F1, LCS F1, table-aware similarity)
- IE: Text coverage + JSON strictness (0.5 * coverage + 0.5 * json_strict)
- json1/json2: DIoU layout score + text score (0.7 * diou + 0.3 * text)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `maritime_ocr_bench` |
| **Dataset ID** | [HiDolphin/MaritimeOCRBench](https://modelscope.cn/datasets/HiDolphin/MaritimeOCRBench/summary) |
| **Paper** | N/A |
| **Tags** | `MultiModal`, `QA` |
| **Metrics** | `score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,888 |
| Prompt Length (Mean) | 102.91 chars |
| Prompt Length (Min/Max) | 23 / 288 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `IE` | 471 | 23 | 23 | 23 |
| `VQA` | 471 | 39.65 | 28 | 141 |
| `parsing` | 472 | 80 | 80 | 80 |
| `json1` | 237 | 256.1 | 248 | 288 |
| `json2` | 237 | 279.9 | 248 | 288 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,888 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 108x50 - 4030x4075 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `IE`

```json
{
  "input": [
    {
      "id": "1d6ce119",
      "content": [
        {
          "text": "请提取所有关键信息，并以 JSON 格式返回。"
        },
        {
          "image": "[BASE64_IMAGE: png, ~550.9KB]"
        }
      ]
    }
  ],
  "target": "{\n  \"Document ID\": \"WiCE Error Message Description\",\n  \"Revision\": \"REV 1\",\n  \"Date\": \"2024-09-12\",\n  \"Company_Logo_Text\":\"WIN GD\",\n  \"Error Messages\": [\n    {\n      \"ID Number\": \"COFD-52\",\n      \"Designation\": \"Fuel Pump Control Signal #2 Fa ... [TRUNCATED 1668 chars] ... nt must be within 4 ~ 20mA.\\n• If necessary, the sensor can be replaced with new one (Caution: before dismantling, do depressurize rail)\"\n    }\n  ],\n  \"Footer\": \"T_PC-Drawing_Portrait | Release: 3.10 (2024-05-15)\",\n  \"Page\": \"Page 20 of 83\"\n}",
  "id": 0,
  "group_id": 0,
  "subset_key": "IE",
  "metadata": {
    "task_type": "IE",
    "prompt": "<image>请提取所有关键信息，并以 JSON 格式返回。",
    "images": [
      "images/580968569085497344_f33f02a81a.png"
    ]
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets maritime_ocr_bench \
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
    datasets=['maritime_ocr_bench'],
    dataset_args={
        'maritime_ocr_bench': {
            # subset_list: ['IE', 'VQA', 'parsing']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


