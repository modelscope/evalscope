# OmniDocBench-v1.6


## Overview

OmniDocBench v1.6 evaluates end-to-end document parsing for text, formulas, tables, layout, and reading order. This adapter is intentionally restricted to the official v1.6 data and scoring contract.

## Task Description

- **Task Type**: End-to-end document parsing
- **Input**: A complete document page image
- **Output**: Markdown containing the page text, formulas, tables, and reading order
- **Domain**: Multilingual academic, financial, textbook, newspaper, magazine, and presentation documents

## Key Features

- Uses `OpenDataLab/OmniDocBench` at the pinned ModelScope revision `297ee5063d6ecc36fe14f3eb4f456607cc895f4a`
- Contains 1,651 pages: 1,355 base pages plus 100 equation-hard, 99 layout-hard, and 97 table-hard pages
- Accepts only the verified v1.6 annotation and rejects other releases and the legacy TSV format
- Scores each page independently with the official v1.6 evaluator in a reusable ms-enclave Docker sandbox

## Evaluation Notes

- Uses MGAM `quick_match`, formula CDM, table TEDS/TEDS-S, edit distance, and reading-order evaluation.
- EvalScope averages page metrics and computes Overall only from the aggregated text, formula, and table components.
- Edit-distance metrics use the 0-1 scale.
- CDM, TEDS, TEDS-S, and Overall use the 0-100 scale.
- Docker with amd64 support and `evalscope[sandbox]` are required.
- The default image is pinned; custom image overrides are allowed, but incompatible images fail during scoring.
- The sandbox pool defaults to one container; increase `sandbox.pool_size` only when sufficient memory is available.
- The official image is large; ensure sufficient disk and memory before evaluation.
- Scores are not directly comparable with the legacy `omni_doc_bench` integration.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `omni_doc_bench_v1_6` |
| **Dataset ID** | [OpenDataLab/OmniDocBench](https://modelscope.cn/datasets/OpenDataLab/OmniDocBench/summary) |
| **Paper** | [Paper](https://github.com/opendatalab/OmniDocBench) |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `text_block_Edit_dist`, `display_formula_Edit_dist`, `display_formula_CDM`, `table_TEDS`, `table_TEDS_structure_only`, `table_Edit_dist`, `reading_order_Edit_dist`, `overall` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,651 |
| Prompt Length (Mean) | 1408 chars |
| Prompt Length (Min/Max) | 1408 / 1408 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,651 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 570x829 - 10142x14342 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d1436bbb",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~433.3KB]"
        },
        {
          "text": " You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:\n\n    1. Text Processing:\n    - Accurately recognize all text content in the PDF image without guessing or i ... [TRUNCATED 924 chars] ... sible.\n\n    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.\n"
        }
      ]
    }
  ],
  "target": "{\"layout_dets\": [{\"category_type\": \"text_block\", \"poly\": [268.9431, 319.97520000000003, 322.9962, 319.97520000000003, 322.9962, 351.0839, 268.9431, 351.0839], \"ignore\": false, \"order\": 2, \"anno_id\": \"box_id_0\", \"attribute\": {}, \"text\": \"that\" ... [TRUNCATED 7763 chars] ... th\": 1653, \"image_path\": \"page-d1561665-5359-42fe-920c-d6e3bff81953.png\", \"page_attribute\": {\"data_source\": \"book\", \"language\": \"english\", \"layout\": \"single_column\", \"special_issue\": [], \"subset\": \"equation_hard\"}}, \"extra\": {\"relation\": []}}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "omnidocbench_version": "v1.6",
    "dataset_revision": "297ee5063d6ecc36fe14f3eb4f456607cc895f4a",
    "image_name": "page-d1561665-5359-42fe-920c-d6e3bff81953.png"
  }
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

## Sandbox Configuration

This benchmark requires a sandbox environment for code execution.

```json
{
  "image": "ghcr.io/zeng-weijun/omnidocbench-eval@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617",
  "entrypoint": [],
  "command": [
    "sleep",
    "infinity"
  ],
  "platform": "linux/amd64",
  "working_dir": "/workspace",
  "network_enabled": false,
  "tools_config": {
    "python_executor": {}
  }
}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_doc_bench_v1_6 \
    --sandbox '{"enabled": true}' \
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
    datasets=['omni_doc_bench_v1_6'],
    sandbox={'enabled': True},
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
