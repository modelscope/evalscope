# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetHub, Sample
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


# The judge returns CORRECT or INCORRECT in the JSON verdict field.
class Grade(BaseModel):
    reasoning: str = ''
    verdict: Literal['CORRECT', 'INCORRECT']


GRADE_CONTRACT = OutputContract(schema_model=Grade)

# Default judge prompt template
JUDGE_PROMPT = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT. For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {correct_answer}
CANDIDATE ANSWER TO ASSESS: {response}

Set the JSON verdict field to CORRECT or INCORRECT."""

PROMPT_TEMPLATE = """
BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

DEFAULT_CACHE_SUBDIR = 'aa_lcr'
DOCUMENT_ARCHIVE_PATH = 'extracted_text/AA-LCR_extracted-text.zip'
DEFAULT_ZIP_NAME = 'AA-LCR_extracted-text.zip'
DEFAULT_EXTRACTED_DIR_NAME: str = 'lcr'


def _read_document(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError as original_error:
        # The official ZIP has UTF-8 filenames without the UTF-8 flag, including
        # an NFD name. Support existing zipfile extractions as well as Unicode names.
        for form in ('NFC', 'NFD'):
            filename = unicodedata.normalize(form, file_path.name)
            for name in (filename, filename.encode('utf-8').decode('cp437')):
                if name == file_path.name:
                    continue
                try:
                    return file_path.with_name(name).read_text(encoding='utf-8')
                except FileNotFoundError:
                    continue
        raise original_error


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr',
        evaluation_version='v1.1',
        pretty_name='AA-LCR',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

AA-LCR (Artificial Analysis Long Context Retrieval) is a benchmark for evaluating long-context retrieval and reasoning capabilities of language models. It requires models to find and synthesize information across multiple documents.

## Task Description

- **Task Type**: Long-Context Question Answering
- **Input**: Multiple documents + question requiring cross-document reasoning
- **Output**: Answer synthesized from document information
- **Context**: Very long context (multiple documents concatenated)

## Key Features

- Tests long-context retrieval abilities
- Multiple document understanding
- Cross-document reasoning required
- LLM-based judging for answer correctness
- Auto-download of document corpus

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy** (via LLM judge)
- Evaluates on **test** split
- Documents auto-downloaded if `text_dir` not specified
- Judge prompt compares candidate answer against reference
""",  # noqa: E501
        dataset_id='evalscope/AA-LCR',
        hf_dataset_id='ArtificialAnalysis/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files; if null will auto-download & extract.',
                'value': None,
            }
        },
    )
)
class AALCRAdapter(DefaultDataAdapter):
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get extra parameters
        self.text_dir = self.extra_params.get('text_dir')

    def load(self):
        # Auto download and extract when text_dir is not provided
        if not self.text_dir:
            self.text_dir = self._ensure_text_dir_downloaded()
        elif not Path(self.text_dir).exists():
            raise ValueError(
                'AA-LCR text_dir does not exist: '
                f'{self.text_dir}. Please provide a valid directory or omit text_dir to auto-download.'
            )

        self.text_dir = Path(self.text_dir)
        return super().load()

    def _ensure_text_dir_downloaded(self) -> Path:
        """Ensure AA-LCR extracted texts are available locally; download and extract if missing."""
        cache_root = Path(DEFAULT_EVALSCOPE_CACHE_DIR) / DEFAULT_CACHE_SUBDIR
        extracted_dir = cache_root / DEFAULT_EXTRACTED_DIR_NAME

        if extracted_dir.exists():
            logger.info(f'AA-LCR documents found: {extracted_dir}')
            return extracted_dir

        cache_root.mkdir(parents=True, exist_ok=True)
        zip_path = cache_root / DEFAULT_ZIP_NAME

        try:
            archive = DatasetHub(
                data_id_or_path=self.dataset_id,
                data_source=self.dataset_hub,
                revision=self.dataset_revision,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            ).download_file(DOCUMENT_ARCHIVE_PATH)
            logger.info(f'Copying AA-LCR documents from {archive} to {zip_path}...')
            zip_path.write_bytes(Path(archive).read_bytes())

            logger.info(f'Extracting {zip_path} to {cache_root}...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_root)

            if not extracted_dir.exists():
                raise ValueError(f'Extraction succeeded but target directory not found: {extracted_dir}')

            logger.info(f'AA-LCR documents ready at {extracted_dir}')
            return extracted_dir
        except Exception as e:
            raise ValueError(
                f'Failed to download or extract AA-LCR documents: {e}. '
                'You can also manually download and set extra_params["text_dir"].'
            ) from e
        finally:
            # Best-effort cleanup of the zip file
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except Exception:
                pass

    def _get_context(self, record: Dict[str, Any]) -> str:
        doc_folder = self.text_dir / record['document_category'] / record['document_set_id']
        manifest = record.get('data_source_filenames')
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError(
                f'AA-LCR data_source_filenames must be a non-empty semicolon-separated string: {doc_folder}'
            )
        filenames = manifest.split(';')
        doc_blocks = []
        # The official per-question manifest defines both membership and order.
        for filename in filenames:
            if not filename.strip():
                raise ValueError(f'AA-LCR data_source_filenames contains an empty filename: {doc_folder}')
            if filename in ('.', '..') or Path(filename).name != filename:
                raise ValueError(f'AA-LCR data_source_filenames must contain filenames without paths: {doc_folder}')
            file_path = doc_folder / filename
            try:
                doc_blocks.append(_read_document(file_path).strip())
            except (OSError, UnicodeDecodeError) as e:
                raise ValueError(f'Could not read required AA-LCR document {file_path}: {e}') from e

        documents_text = '\n\n'.join(
            f'BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}' for i, doc in enumerate(doc_blocks)
        )
        return documents_text

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a record to a Sample with long-context prompt."""
        context = self._get_context(record)
        prompt = self.prompt_template.format(documents_text=context, question=record['question'])

        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target=record['answer'],
            metadata={
                'question': record['question'],
                'data_source_urls': record['data_source_urls'],
                'input_tokens': record.get('input_tokens', 0),
            },
        )

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            prompt = (
                JUDGE_PROMPT.format(
                    question=judge_context.task_state.metadata['question'],
                    correct_answer=judge_context.reference,
                    response=judge_context.filtered_prediction,
                )
                + case.output_contract.instruction()
            )
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            return ReducedVerdict(value={'acc': 1.0 if case_verdicts[0].value.verdict == 'CORRECT' else 0.0})

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='grade', output_contract=GRADE_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='acc',
        )
