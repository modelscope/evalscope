# OmniDocBench


## Overview

This adapter preserves EvalScope's original 981-page OmniDocBench TSV integration for compatibility with existing evaluations.

## Task Description

- **Task Type**: Document Parsing and Understanding
- **Input**: PDF page image
- **Output**: Parsed document structure in Markdown format
- **Domain**: Document understanding, OCR, layout analysis

## Key Features

- Uses the legacy `evalscope/OmniDocBench_tsv` dataset with 981 PDF pages
- Covers text blocks, formulas, tables, and reading order
- Keeps the existing local Python scoring implementation and metric names unchanged
- Remains available for reproducing existing EvalScope results

## Evaluation Notes

- This legacy TSV dataset is not labeled as a specific upstream OmniDocBench release.
- For new evaluations, use the recommended `omni_doc_bench_v1_6` benchmark.
- Implements the existing `end2end` and `quick_match` scoring paths.
- Metrics: Edit_dist, BLEU, METEOR (text), TEDS (tables)
- Install the `evalscope[omnidoc_bench]` extra for legacy scoring dependencies.
- Output format: Markdown with LaTeX formulas and HTML tables
- Scores from this legacy integration are not directly comparable with v1.6 scores.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `omni_doc_bench` |
| **Dataset ID** | [evalscope/OmniDocBench_tsv](https://modelscope.cn/datasets/evalscope/OmniDocBench_tsv/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `text_block`, `display_formula`, `table`, `reading_order`, `normalized_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 981 |
| Prompt Length (Mean) | 1408 chars |
| Prompt Length (Min/Max) | 1408 / 1408 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 981 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 516x729 - 10142x14342 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "7c6fda98",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~321.8KB]"
        },
        {
          "text": " You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:\n\n    1. Text Processing:\n    - Accurately recognize all text content in the PDF image without guessing or i ... [TRUNCATED 924 chars] ... sible.\n\n    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.\n"
        }
      ]
    }
  ],
  "target": "{\"layout_dets\": [{\"category_type\": \"title\", \"poly\": [102.5999912116609, 120.87255879760278, 719.3118659856144, 120.87255879760278, 719.3118659856144, 194.14083813380114, 102.5999912116609, 194.14083813380114], \"ignore\": false, \"order\": 1, \"an ... [TRUNCATED 9876 chars] ... nguage\": \"simplified_chinese\", \"layout\": \"1andmore_column\", \"special_issue\": [\"watermark\"]}, \"page_no\": 11, \"height\": 1500, \"width\": 2667, \"image_path\": \"eastmoney_59cde7e939acc3124df9d3f2c85b5a0ec41b9da1157d5be38e098672022b47cb.pdf_11.jpg\"}}",
  "id": 0,
  "group_id": 0
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
 You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

    1. Text Processing:
    - Accurately recognize all text content in the PDF image without guessing or inferring.
    - Convert the recognized text into Markdown format.
    - Maintain the original document structure, including headings, paragraphs, lists, etc.

    2. Mathematical Formula Processing:
    - Convert all mathematical formulas to LaTeX format.
    - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
    - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

    3. Table Processing:
    - Convert tables to HTML format.
    - Wrap the entire table with <table> and </table>.

    4. Figure Handling:
    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

    5. Output Format:
    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
    - For complex layouts, try to maintain the original document's structure and format as closely as possible.

    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.

```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `match_method` | `str` | `quick_match` | Scoring match method used for evaluation. Choices: ['quick_match', 'simple_match', 'no_split'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_doc_bench \
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
    datasets=['omni_doc_bench'],
    dataset_args={
        'omni_doc_bench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
