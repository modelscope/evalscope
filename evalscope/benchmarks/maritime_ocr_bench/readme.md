
数据集地址：https://www.modelscope.cn/datasets/HiDolphin/MaritimeOCRBench

## Overview

Maritime-OCR-Bench is a multimodal OCR and document understanding benchmark built for maritime and document-centric evaluation. In the maritime portion of the dataset, samples include container markings, bills of lading, shipping forms, port and cargo documents, and ship or engine manuals. The released benchmark also covers broader real-world document scenarios such as receipts, invoices, reports, scanned pages, tables, resumes, presentations, and scene-style text images.

The dataset is organized as an instruction-following multimodal conversation set. Each sample contains one image path, a user instruction, a reference answer, and a task identifier. The benchmark is designed to cover both free-form understanding and strict structured output:

- VQA and Parsing focus on semantic understanding, reading order, and long-text reconstruction.
- IE focuses on strict JSON compliance and field-level extraction quality.
- JSON Spotting focuses on text content recognition together with layout coordinates and output-schema stability.

The released public test split contains 1888 samples across 5 task types.

## Task Types

| Task | task_type in dataset | Samples | Description |
| --- | --- | ---: | --- |
| IE | `IE` | 471 | Strict information extraction with JSON-formatted answers. |
| VQA | `vqa` | 471 | Visual question answering over documents, forms, tables, receipts, and scene text. |
| Parsing | `parsing` | 472 | OCR parsing and full-text reconstruction, including long structured content. |
| JSON Spotting v1 | `json1` | 237 | Text spotting with page-oriented JSON output. |
| JSON Spotting v2 | `json2` | 237 | Text spotting with unified `images`-based JSON output. |

Inside evalscope, the benchmark adapter maps these task types into five report subsets: `IE`, `VQA`, `parsing`, `json1`, and `json2`.

## Data Format

The dataset annotation file uses JSONL format. Each line is one sample with the following top-level fields:

- `images`: relative image path list.
- `messages`: two-turn conversation containing the user instruction and the reference answer.
- `task_type`: task identifier used to choose the scoring logic.

Example:

```json
{
    "messages": [
        {
            "role": "user",
            "content": "<image>Please extract all key information and return it in JSON format."
        },
        {
            "role": "assistant",
            "content": "{\n  \"field_a\": \"value\"\n}"
        }
    ],
    "images": [
        "images/example_image.jpg"
    ],
    "task_type": "IE"
}
```

## File Structure

When downloaded from ModelScope or prepared locally, the dataset is expected to follow this structure:

```text
MaritimeOCRBench/
|- README.md
|- README.zh-CN.md
|- assets/
|- maritime_ocr_bench.jsonl
`- images/
```

For evalscope evaluation, the adapter supports both:

- a local dataset path pointing to the JSONL file directly
- a local directory or ModelScope snapshot containing `maritime_ocr_bench.jsonl` and `images/`

## Evaluation Protocol

This benchmark uses task-specific metrics rather than a single unified text metric.

### VQA / Parsing

VQA and Parsing use a composite text score. Before scoring, text is normalized and compared from multiple views including normalized text, compact text, loose text, and table-aware text. The final score is a weighted combination of edit-distance similarity, character-level F1, LCS F1, loose edit-distance similarity, and table-aware similarity.

### IE

IE uses a combined score of text coverage and strict JSON validity:

- text coverage measures how much reference information is preserved in the model output
- strict JSON validity checks whether the output is valid JSON without extra explanation or Markdown fences

The final IE score is:

```text
ie_score = 0.5 * text_coverage_score + 0.5 * json_strict_score
```

### JSON Spotting v1 / v2

Spotting tasks combine layout localization quality and text recognition quality. The final score is:

```text
overall_score = 0.7 * diou_score + 0.3 * text_score
```

The `json1` and `json2` subsets differ mainly in expected output schema.

## Usage In evalscope

Example:

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
        model='qwen-vl-max',
        api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key='',
        datasets=['maritime_ocr_bench'],
        eval_batch_size=20,
        generation_config={
                'retries': 1,
                'retry_interval': 1,
                'timeout': 300,
        },
        ignore_errors=True,
        limit=None,
        debug=False,
)

run_task(task_cfg=task_cfg)
```

If you want to run against a local copy of the dataset, point `dataset_id` or the task configuration to the local JSONL file or dataset directory prepared from the public release.

## Notes

- The benchmark is evaluated on the released public test split.
- The adapter resolves image paths relative to the actual local dataset root or downloaded ModelScope snapshot.
- Reports in evalscope can be aggregated by task subset: `IE`, `VQA`, `parsing`, `json1`, and `json2`.
- This document intentionally excludes model leaderboard or reference result tables.
