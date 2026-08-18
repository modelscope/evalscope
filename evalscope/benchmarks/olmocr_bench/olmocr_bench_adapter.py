# olmOCR-Bench adapter: unit-test style evaluation of end-to-end PDF-to-Markdown transcription.
#
# Dataset: https://huggingface.co/datasets/allenai/olmOCR-bench (ODC-BY-1.0)
# Official evaluation code: https://github.com/allenai/olmocr (Apache-2.0)

import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DataLoader, Dataset, DictDataLoader, Sample, download_dataset_file
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger
from .unit_tests import load_single_test

logger = get_logger()

# Data is only published on the Hugging Face hub; there is no ModelScope mirror.
DATASET_HUB = HubType.HUGGINGFACE

# Sources whose rules are entirely math type; excluded until KaTeX-rendered comparison lands.
UNSUPPORTED_SUBSETS = ['arxiv_math', 'old_scans_math']

SUBSET_LIST = ['headers_footers', 'long_tiny_text', 'multi_column', 'old_scans', 'table_tests']

# Official no-document-anchoring transcription prompt
# (olmocr/bench/prompts.py, build_openai_silver_data_prompt_no_document_anchoring).
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
"""


@register_benchmark(
    BenchmarkMeta(
        name='olmocr_bench',
        pretty_name='olmOCR-Bench',
        tags=[Tags.MULTI_MODAL, Tags.QA],
        description=DESCRIPTION,
        dataset_id='allenai/olmOCR-bench',
        paper_url='https://arxiv.org/abs/2502.18443',
        metric_list=['pass_rate'],
        primary_metric='pass_rate',
        eval_split='test',
        subset_list=SUBSET_LIST,
        prompt_template=PROMPT_TEMPLATE,
    )
)
class OlmocrBenchAdapter(VisionLanguageAdapter):
    """Data adapter for allenai/olmOCR-bench.

    Unit tests are grouped by (pdf, page) so each sample transcribes one page once; the grouped
    rules are carried in sample metadata and replayed against the transcription in match_score.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.add_overall_metric = True

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Load one source's unit-test JSONL and group it into per-page records."""
        if subset in UNSUPPORTED_SUBSETS:
            raise ValueError(
                f"Subset '{subset}' contains math rules only, which require KaTeX rendering and are "
                f'not supported. Valid subsets are: {SUBSET_LIST}'
            )

        jsonl_path = Path(
            download_dataset_file(
                data_id_or_path=self.dataset_id,
                file_path=f'bench_data/{subset}.jsonl',
                data_source=DATASET_HUB,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            )
        )

        tests_by_page: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for line in jsonl_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            test = json.loads(line)
            tests_by_page[(test['pdf'], test['page'])].append(test)

        page_records = [{
            'pdf': pdf,
            'page': page,
            'tests': tests,
        } for (pdf, page), tests in tests_by_page.items()]

        return DictDataLoader(
            dict_list=page_records,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            shuffle_choices=self.shuffle_choices,
            seed=self.seed,
        ).load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Render one PDF page and attach its unit tests as metadata."""
        pdf_path = Path(
            download_dataset_file(
                data_id_or_path=self.dataset_id,
                file_path=f"bench_data/pdfs/{record['pdf']}",
                data_source=DATASET_HUB,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            )
        )
        image_b64 = self._render_pdf_page(pdf_path, record['page'])
        content: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=self.prompt_template),
        ]
        return Sample(
            input=[ChatMessageUser(content=content)],
            target=f"{record['pdf']}#page={record['page']}",
            metadata={
                'pdf': record['pdf'],
                'page': record['page'],
                'tests': record['tests'],
            },
        )

    def _render_pdf_page(self, pdf_path: Path, page_number: int, dpi: int = 150) -> str:
        """Render a 1-based PDF page to a base64 JPEG data-URI at the given DPI."""
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        try:
            page = document[page_number - 1]
            bitmap = page.render(scale=dpi / 72.0)
            pil_image = bitmap.to_pil().convert('RGB')
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=90)
            return self._image_bytes_to_base64(buffer.getvalue(), default_format='jpeg')
        finally:
            document.close()

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Keep the full transcription; map a bare `null` reply to an empty transcription.

        The official harness stores `natural_text=null` as an empty .md file, so a model that
        follows the prompt and answers `null` for a blank page must be scored against an empty
        transcription, not the literal string "null".
        """
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
                # A load failure means the dataset row is malformed; fail fast instead of silently
                # scoring it as a model mistake.
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
        """Aggregate as the fraction of unit tests that pass (official per-source metric).

        `num` counts unit tests (and doubles as the aggregation weight), while `ids` lists
        page-level sample ids; the two intentionally differ.
        """
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
                num=tests_total,
                ids=ids,
                metadata={
                    'tests_passed': tests_passed,
                    'tests_total': tests_total
                },
            )
        ]
