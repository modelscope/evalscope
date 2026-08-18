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

- 1,403 PDFs and 7,010 unit tests from the official release; each test states one property a
  correct transcription must satisfy (text present, text absent, reading order, table structure,
  or a baseline sanity check)
- Scoring rules are ported 1:1 from the official `olmocr` bench implementation, so per-subset
  scores are directly comparable with the official report
- This adapter covers the five non-math sources (`headers_footers`, `long_tiny_text`,
  `multi_column`, `old_scans`, `table_tests`), 845 pages and 3,634 unit tests; the two math-only
  sources (`arxiv_math`, `old_scans_math`) require KaTeX-rendered equation comparison and are not
  included
- Pages are rendered with `pypdfium2` at 150 DPI; models receive one image per sample

## Evaluation Notes

- Each sample is one PDF page; its unit tests are evaluated against the model transcription and
  the subset score is the fraction of unit tests that pass, matching the official per-source
  metric
- The primary metric is `pass_rate`; each subset score equals the official per-source pass rate.
  The report's overall score pools all unit tests across subsets (weighted by test count), which
  differs slightly from the official total (an unweighted mean of the per-JSONL-file scores);
  compare per-subset scores for exact parity with the official report
- Fuzzy matching thresholds (`max_diffs`), positional constraints (`first_n`/`last_n`), case
  sensitivity, and table relationship checks follow the official implementation exactly
- A bare `null` reply is treated as an empty transcription, mirroring how the official harness
  stores `natural_text=null` as an empty file
- Requires `pip install evalscope[olmocr_bench]` (rapidfuzz, fuzzysearch, beautifulsoup4,
  pypdfium2); the dataset is downloaded from Hugging Face
- [Paper](https://arxiv.org/abs/2502.18443) | [Code](https://github.com/allenai/olmocr) |
  [Dataset](https://huggingface.co/datasets/allenai/olmOCR-bench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `olmocr_bench` |
| **Dataset ID** | [allenai/olmOCR-bench](https://modelscope.cn/datasets/allenai/olmOCR-bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2502.18443) |
| **Tags** | `MultiModal`, `QA` |
| **Metrics** | `pass_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

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
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
