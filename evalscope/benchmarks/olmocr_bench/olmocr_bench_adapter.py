# olmOCR-Bench adapter: unit-test style evaluation of end-to-end document-to-Markdown transcription.
#
# Dataset: https://modelscope.cn/datasets/evalscope/olmOCR-Bench (ODC-BY-1.0), a pre-rendered
# page-image mirror of https://huggingface.co/datasets/allenai/olmOCR-bench.
# Official evaluation code: https://github.com/allenai/olmocr (Apache-2.0)

import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from .unit_tests import load_single_test

logger = get_logger()

# Math-only sources (arxiv_math, old_scans_math) are excluded: their rules need KaTeX rendering.
SUBSET_LIST = ['headers_footers', 'long_tiny_text', 'multi_column', 'old_scans', 'table_tests']

# Official prompt (olmocr/bench/prompts.py, build_openai_silver_data_prompt_no_document_anchoring).
PROMPT_TEMPLATE = (
    'Below is the image of one page of a PDF document. '
    'Just return the plain text representation of this document as if you were reading it naturally.\n'
    'Turn equations into a LaTeX representation, and tables into markdown format. '
    'Remove the headers and footers, but keep references and footnotes.\n'
    'Read any natural handwriting.\n'
    'This is likely one page out of several in the document, so be sure to preserve any sentences '
    'that come from the previous page, or continue onto the next page, exactly as they are.\n'
    'If there is no text at all that you think you should read, you can output null.\n'
    'Do not hallucinate.'
)  # noqa: E501

DESCRIPTION = """
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
"""


@register_benchmark(
    BenchmarkMeta(
        name='olmocr_bench',
        pretty_name='olmOCR-Bench',
        tags=[Tags.MULTI_MODAL, Tags.QA],
        description=DESCRIPTION,
        dataset_id='evalscope/olmOCR-Bench',
        paper_url='https://arxiv.org/abs/2502.18443',
        metric_list=['pass_rate'],
        primary_metric='pass_rate',
        eval_split='test',
        subset_list=SUBSET_LIST,
        prompt_template=PROMPT_TEMPLATE,
    )
)
class OlmocrBenchAdapter(VisionLanguageAdapter):
    """Data adapter for evalscope/olmOCR-Bench.

    The dataset is a single parquet loaded through the standard remote flow: one row per
    ``(pdf, page)`` carrying the pre-rendered page image, the ``subset`` key, and the page's unit
    tests (a JSON string). Rows are split into subsets by ``subset_key`` and the grouped rules are
    replayed against the transcription in ``match_score``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Build a sample from a parquet row: pre-rendered image + this page's unit tests."""
        content: List[Content] = [ContentText(text=self.prompt_template)]
        image = record.get('image')
        if image and image.get('bytes'):
            content.append(ContentImage(image=bytes_to_base64(image['bytes'], format='png', add_header=True)))
        return Sample(
            input=[ChatMessageUser(content=content)],
            target=f"{record['pdf']}#page={record['page']}",
            subset_key=record['subset'],
            metadata={
                'pdf': record['pdf'],
                'page': record['page'],
                'tests': json.loads(record['tests']),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Map a bare `null` reply to empty transcription (official harness stores it as empty .md)."""
        if prediction is None:
            return ''
        if prediction.strip().lower() == 'null':
            return ''
        return prediction.strip()

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Replay this page's unit tests against the transcription."""
        tests = task_state.metadata.get('tests', [])
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)

        if not tests:
            score.value = {'pass_rate': 0.0, 'tests_passed': 0, 'tests_total': 0}
            score.main_score_name = 'pass_rate'
            return score

        passed = 0
        failures = []
        for test_data in tests:
            try:
                test = load_single_test(test_data)
            except Exception as e:
                raise ValueError(
                    f'Failed to load unit test {test_data.get("id")!r}; the dataset JSONL row may be malformed'
                ) from e
            try:
                test_passed, explanation = test.run(filtered_prediction)
            except Exception as e:
                logger.error(f'Error running unit test {test_data.get("id")}: {e}')
                test_passed, explanation = False, f'error: {e}'
            if test_passed:
                passed += 1
            else:
                failures.append(f'{test_data.get("id")}: {explanation}')

        score.value = {
            'pass_rate': passed / len(tests),
            'tests_passed': float(passed),
            'tests_total': float(len(tests)),
        }
        score.main_score_name = 'pass_rate'
        score.metadata['failed_tests'] = failures
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Pooled pass rate over all unit tests in the subset (official per-source metric)."""
        tests_passed = 0
        tests_total = 0
        ids = []
        for sample_score in sample_scores:
            value = sample_score.score.value
            tests_passed += int(value.get('tests_passed', 0.0))
            tests_total += int(value.get('tests_total', 0.0))
            ids.append(sample_score.sample_id)

        pass_rate = tests_passed / tests_total if tests_total > 0 else 0.0
        return [
            AggScore(
                score=pass_rate,
                metric_name='pass_rate',
                aggregation='unit_test_pass_rate',
                num=len(sample_scores),
                ids=ids,
                metadata={
                    'tests_passed': tests_passed,
                    'tests_total': tests_total
                },
            )
        ]
