# OmniDocBench-v1.6


## Overview

OmniDocBench v1.6 evaluates end-to-end document parsing for text, formulas, tables, layout, and reading order. This adapter is intentionally restricted to the official v1.6 data and scoring contract.

## Version and Data Source

- **Benchmark**: `omni_doc_bench_v1_6`
- **Dataset**: `OpenDataLab/OmniDocBench`, pinned to ModelScope revision `297ee5063d6ecc36fe14f3eb4f456607cc895f4a`
- **Scale**: 1,651 pages, including the 1,355-page v1.5 set and 296 equation, layout, and table hard pages
- **Compatibility**: other OmniDocBench releases and the legacy TSV integration are rejected

## Evaluation

Each page is scored independently by the official v1.6 evaluator inside an ms-enclave Docker sandbox. The sandbox reuses the pinned official image and runs MGAM `quick_match`, formula CDM, table TEDS/TEDS-S, edit distance, and reading-order evaluation. EvalScope averages the official page metrics and computes Overall only after all pages are aggregated.

- Edit-distance metrics use the 0-1 scale.
- CDM, TEDS, TEDS-S, and Overall use the 0-100 scale.
- Docker with amd64 support and `evalscope[sandbox]` are required.
- The sandbox pool defaults to one container; increase `sandbox.pool_size` only when sufficient memory is available.
- The official image is large; ensure sufficient disk and memory before evaluation.
- Scores are not directly comparable with the legacy `omni_doc_bench` v1.5 integration.


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
      "id": "46712ccc",
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
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "omnidocbench_version": "v1.6",
    "dataset_revision": "297ee5063d6ecc36fe14f3eb4f456607cc895f4a",
    "annotation_sha256": "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496",
    "image_name": "page-d1561665-5359-42fe-920c-d6e3bff81953.png",
    "annotation": {
      "layout_dets": [
        {
          "category_type": "text_block",
          "poly": [
            268.9431,
            319.97520000000003,
            322.9962,
            319.97520000000003,
            322.9962,
            351.0839,
            268.9431,
            351.0839
          ],
          "ignore": false,
          "order": 2,
          "anno_id": "box_id_0",
          "attribute": {},
          "text": "that"
        },
        {
          "category_type": "equation_isolated",
          "poly": [
            404.98500000000007,
            362.07719999999995,
            816.9871826171875,
            362.07719999999995,
            816.9871826171875,
            448.3244323730468,
            404.98500000000007,
            448.3244323730468
          ],
          "ignore": false,
          "order": 3,
          "anno_id": "box_id_1",
          "attribute": {},
          "latex": "$$AB = \n\\left[\\begin{array}{ccc}\n2 & 3 \\\\\n1 & 4\n\\end{array}\\right]\n\\left[\\begin{array}{ccc}\n5 & 2 & 1 \\\\\n3 & 8 & 6\n\\end{array}\\right]$$"
        },
        {
          "category_type": "text_block",
          "poly": [
            271.9185,
            551.0684,
            1379.9243999999999,
            551.0684,
            1379.9243999999999,
            698.8932,
            271.9185,
            698.8932
          ],
          "ignore": false,
          "order": 5,
          "anno_id": "box_id_2",
          "attribute": {},
          "text": "When an attempt is made to form the product $ \\mathbf{{BA}} $ , we discover that the dimensions are not compatible in this order because the rows of $ \\mathbf{B} $ are three-dimensional vectors and the columns of $ \\mathbf{A} $ are two-dimensional vectors. Hence the dot product of the $ j $ th row of and the $ k $ th column of $ \\mathbf{A} $ is not defined.$\\blacksquare$"
        },
        {
          "category_type": "text_block",
          "poly": [
            274.0674,
            718.073,
            1374.9654,
            718.073,
            1374.9654,
            795.0260999999999,
            274.0674,
            795.0260999999999
          ],
          "ignore": false,
          "order": 6,
          "anno_id": "box_id_3",
          "attribute": {},
          "text": "If it happens that $ \\mathbf{{AB}} = \\mathbf{{BA}} $ , we say that $ \\mathbf{A} $ and $ \\mathbf{B} $ commute. Most often,even when $ \\mathbf{{AB}} $ and $ \\mathbf{{BA}} $ are both defined, the products are not necessarily the same."
        },
        {
          "category_type": "text_block",
          "poly": [
            277.0428,
            799.0024000000001,
            1379.9243999999999,
            799.0024000000001,
            1379.9243999999999,
            1025.8854,
            277.0428,
            1025.8854
          ],
          "ignore": false,
          "order": 7,
          "anno_id": "box_id_4",
          "attribute": {},
          "text": "We now discuss how to use matrices to represent a linear system of equations. The linear equations in (3) can be written as a matrix product. The coefficients $a _ { k j }$ are stored in a matrix $\\pmb { A }$ (called the coefficient matrix) o ... [TRUNCATED 73 chars] ... atrix $\\pmb { X }$ of dimension $N \\times 1$ .The constants $\\boldsymbol { b } _ { k }$ are stored in a matrix $\\pmb { B }$ of dimension $M \\times 1$ . It is conventional to use column matrices for both $\\pmb { X }$ and $\\pmb { B }$ and write"
        },
        {
          "category_type": "equation_isolated",
          "poly": [
            284.78271484375,
            1038.919444522659,
            1277.9343000000001,
            1038.919444522659,
            1277.9343000000001,
            1328.019844522659,
            284.78271484375,
            1328.019844522659
          ],
          "ignore": false,
          "order": 8,
          "anno_id": "box_id_5",
          "attribute": {},
          "latex": "$$\\mathbf{A}\\mathbf{X} = \\left\\lbrack  \\begin{array}{cccccc} {a}_{11} & {a}_{12} & \\cdots & {a}_{1j} & \\cdots & {a}_{1N} \\\\  {a}_{21} & {a}_{22} & \\cdots & {a}_{2j} & \\cdots & {a}_{2N} \\\\  \\vdots & \\vdots & & \\vdots & & \\vdots \\\\  {a}_{k1} &  ... [TRUNCATED 221 chars] ... {1} \\\\  {x}_{2} \\\\  \\vdots \\\\  {x}_{j} \\\\  \\vdots \\\\  {x}_{N} \\end{array}\\right\\rbrack   = \\left\\lbrack  \\begin{array}{cccccc} {b}_{1} \\\\  {b}_{2} \\\\  \\vdots \\\\  {b}_{j} \\\\  \\vdots \\\\  {b}_{M} \\end{array}\\right\\rbrack   = \\mathbf{B}.\\tag{8}$$"
        },
        {
          "category_type": "text_block",
          "poly": [
            279.0264,
            1336.0368,
            1384.0569,
            1336.0368,
            1384.0569,
            1452.9868,
            279.0264,
            1452.9868
          ],
          "ignore": false,
          "order": 9,
          "anno_id": "box_id_6",
          "attribute": {},
          "text": "The matrix multiplication $ \\mathbf{{AX}} = \\mathbf{B} $ in (8) is reminiscent of the dot product for ordinary vectors, because each element ${b}_{k} $in $ \\mathbf{B} $ is the result obtained by taking the dot product of row $k$in matrix $ \\mathbf{A} $ with the column matrix $ \\mathbf{X} $ ."
        },
        {
          "category_type": "text_block",
          "poly": [
            281.01000000000005,
            1485.0311000000002,
            1385.0487,
            1485.0311000000002,
            1385.0487,
            1562.9198000000001,
            281.01000000000005,
            1562.9198000000001
          ],
          "ignore": false,
          "order": 10,
          "anno_id": "box_id_7",
          "attribute": {},
          "text": "Example 3.6. Express the system of linear equations (5) in Example 3.4 as a matrix product. Use matrix multiplication to verify that ${\\left\\lbrack \\begin{array}{lll} 4 & 3 & 3 \\end{array}\\right\\rbrack }^{\\prime } $ is the solution of (5):"
        },
        {
          "category_type": "equation_isolated",
          "poly": [
            278.9978942871094,
            1579.0589,
            1104.0387,
            1579.0589,
            1104.0387,
            1707.9378,
            278.9978942871094,
            1707.9378
          ],
          "ignore": false,
          "order": 11,
          "anno_id": "box_id_8",
          "attribute": {},
          "latex": "$$\\left[\\begin{array}{cccccc}\n0.125 & 0.200 & 0.400 \\\\\n0.375 & 0.500 & 0.600 \\\\\n0.500 & 0.300 & 0.000\\end{array}\\right]\n\\left[\\begin{array}{cccccc}\nx_1 \\\\\nx_2 \\\\\nx_3\\end{array}\\right]\n=\\left[\\begin{array}{cccccc}2.3 \\\\4.8 \\\\2.9\\end{array}\\right].\\tag{9}$$"
        },
        {
          "category_type": "text_block",
          "poly": [
            282.99359999999996,
            1725.0125,
            1386.0405,
            1725.0125,
            1386.0405,
            1815.9995999999999,
            282.99359999999996,
            1815.9995999999999
          ],
          "ignore": false,
          "order": 12,
          "anno_id": "box_id_9",
          "attribute": {},
          "text": "To verify that $ {\\left\\lbrack \\begin{array}{lll} 4 & 3 & 3 \\end{array}\\right\\rbrack }^{\\prime } $ is the solution of (5), we must show that $A{\\left\\lbrack \\begin{array}{lll} 4 & 3 & 3 \\end{array}\\right\\rbrack }^{\\prime } =  {\\left\\lbrack \\begin{array}{lll} {2.3} & {4.8} & {2.9} \\end{array}\\right\\rbrack }^{\\prime } $ :"
        },
        "... [TRUNCATED 6 more items] ..."
      ],
      "page_info": {
        "page_no": 0,
        "height": 2339,
        "width": 1653,
        "image_path": "page-d1561665-5359-42fe-920c-d6e3bff81953.png",
        "page_attribute": {
          "data_source": "book",
          "language": "english",
          "layout": "single_column",
          "special_issue": [],
          "subset": "equation_hard"
        }
      },
      "extra": {
        "relation": []
      }
    }
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
