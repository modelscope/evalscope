# olmOCR-Bench


## Overview

olmOCR-Bench evaluates end-to-end document transcription: a model reads one rendered PDF page and
returns the full Markdown transcription of that page, which is then checked against human-written
unit tests instead of a single reference answer.

## Task Description

- **Task Type**: PDF page to Markdown transcription
- **Input**: A rendered PDF page image
- **Output**: The complete Markdown transcription of the page
- **Domain**: Academic papers, scanned books, historical scans, and internal documents

## Key Features

- 1,403 PDF pages and 7,019 unit tests in the released bench data; each test states one property
  a correct transcription must satisfy (text present, text absent, reading order, table structure,
  or a baseline sanity check)
- Scoring rules are ported 1:1 from the official `olmocr` bench implementation, so per-subset
  scores are directly comparable with the official report
- This adapter covers the five non-math sources (`headers_footers`, `long_tiny_text`,
  `multi_column`, `old_scans`, `table_tests`), 845 pages and 3,634 unit tests; the two math-only
  sources (`arxiv_math`, `old_scans_math`) require KaTeX-rendered equation comparison and are not
  included
- Pages are pre-rendered once to PNG images (longest side 2048 px, matching the official renderer
  `render_pdf_to_base64png` with `target_longest_image_dim=2048`) and packaged, together with the
  unit tests, into a single parquet on ModelScope; the adapter loads it through the standard remote
  flow (one download, native parsing) with no PDF rasterization at eval time

## Evaluation Notes

- Each sample is one PDF page; its unit tests are evaluated against the model transcription and
  the subset score is the fraction of unit tests that pass, matching the official per-source
  metric
- The primary metric is `pass_rate`. Each subset score equals the official per-source pass rate
  (the fraction of that source's unit tests that pass), so per-subset scores match the official
  report exactly
- The official total is the unweighted mean of the per-source pass rates. It is recorded as the
  report's `macro_score` (per category), while the headline overall score is EvalScope's native
  sample-weighted (per-page) micro average; the two differ unless the subsets have equal page
  counts, so compare per-subset scores or `macro_score` for parity with the official report
- `num` reports the number of PDF pages (samples) per subset, so the overall sample count matches
  the prediction records rather than the unit-test count
- Fuzzy matching thresholds (`max_diffs`), positional constraints (`first_n`/`last_n`), case
  sensitivity, and table relationship checks follow the official implementation exactly
- A bare `null` reply is treated as an empty transcription, mirroring how the official harness
  stores `natural_text=null` as an empty file
- Requires `pip install evalscope[olmocr_bench]` (rapidfuzz, fuzzysearch, beautifulsoup4); the
  benchmark is a single parquet on ModelScope (`evalscope/olmOCR-Bench`, image bytes + unit tests),
  a mirror of the Hugging Face release, loaded natively in one download
- [Paper](https://arxiv.org/abs/2502.18443) | [Code](https://github.com/allenai/olmocr) |
  [Dataset](https://modelscope.cn/datasets/evalscope/olmOCR-Bench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `olmocr_bench` |
| **Dataset ID** | [evalscope/olmOCR-Bench](https://modelscope.cn/datasets/evalscope/olmOCR-Bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2502.18443) |
| **Tags** | `MultiModal`, `QA` |
| **Metrics** | `pass_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 845 |
| Prompt Length (Mean) | 599 chars |
| Prompt Length (Min/Max) | 599 / 599 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `headers_footers` | 266 | 599 | 599 | 599 |
| `long_tiny_text` | 62 | 599 | 599 | 599 |
| `multi_column` | 231 | 599 | 599 | 599 |
| `old_scans` | 98 | 599 | 599 | 599 |
| `table_tests` | 188 | 599 | 599 | 599 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 845 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 2048x773 - 2048x1957 |
| Formats | png |


## Sample Example

**Subset**: `headers_footers`

```json
{
  "input": [
    {
      "id": "0c5d3fad",
      "content": [
        {
          "text": "Below is the image of one page of a PDF document. Just return the plain text representation of this document as if you were reading it naturally.\nTurn equations into a LaTeX representation, and tables into markdown format. Remove the headers  ... [TRUNCATED 115 chars] ... l in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.\nIf there is no text at all that you think you should read, you can output null.\nDo not hallucinate."
        },
        {
          "image": "[BASE64_IMAGE: png, ~297.1KB]"
        }
      ]
    }
  ],
  "target": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf#page=1",
  "id": 0,
  "group_id": 0,
  "subset_key": "headers_footers",
  "metadata": {
    "pdf": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf",
    "page": 1,
    "tests": [
      {
        "pdf": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf",
        "page": 1,
        "id": "0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed_pg1_header_01",
        "type": "absent",
        "max_diffs": 2,
        "checked": "verified",
        "url": "https://webges.uv.es/uvTaeWeb/DescargarCertificadoPublicacion.do?codigo=ANUNCIO-C9-2022-1285",
        "text": "Certificado de publicación disponible en http://fandango.accv.es:8070/fa",
        "case_sensitive": false,
        "first_n": null,
        "last_n": null
      },
      {
        "pdf": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf",
        "page": 1,
        "id": "0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed_pg1_header_02",
        "type": "absent",
        "max_diffs": 4,
        "checked": "verified",
        "url": "https://webges.uv.es/uvTaeWeb/DescargarCertificadoPublicacion.do?codigo=ANUNCIO-C9-2022-1285",
        "text": "Este documento será custodiado por la Agencia de Tecnología y Certificación Electrónica - ISTEC Pista de Ademuz S/N. 46100 Burjassot (Valencia). Tel. 902 482 481 Correo-e: accv@accv.es",
        "case_sensitive": false,
        "first_n": null,
        "last_n": null
      }
    ]
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Below is the image of one page of a PDF document. Just return the plain text representation of this document as if you were reading it naturally.
Turn equations into a LaTeX representation, and tables into markdown format. Remove the headers and footers, but keep references and footnotes.
Read any natural handwriting.
This is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.
If there is no text at all that you think you should read, you can output null.
Do not hallucinate.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets olmocr_bench \
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
    datasets=['olmocr_bench'],
    dataset_args={
        'olmocr_bench': {
            # subset_list: ['headers_footers', 'long_tiny_text', 'multi_column']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
